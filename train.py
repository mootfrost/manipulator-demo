import sys
import modal



image = modal.Image.debian_slim(python_version='3.11').pip_install(
    'torch~=2.5.1',
    'torchvision~=0.20.1',
    'scikit-learn~=1.5.2',
    'tqdm~=4.67.0'
)
app = modal.App('grob_ai_training', image=image)
vol = modal.Volume.from_name('grob_ai')


@app.function(volumes={'/data': vol})
def unzip_dataset():
    import zipfile
    import os
    data_dir = '/data/archive.zip'
    with zipfile.ZipFile(data_dir, 'r') as zip_ref:
        zip_ref.extractall('/data/dataset')
    print(os.listdir('/data/dataset'))
    vol.commit()

 
@app.function(
    volumes={'/data': vol},
    gpu=modal.gpu.A100(count=1),
    timeout=0
)
def train_model(num_epochs=25):
    import torch
    import os
    from torchvision import datasets, models, transforms
    import zipfile
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import f1_score, accuracy_score
    import copy
    import time
    from tqdm import tqdm
    vol.reload()
    data_dir = '/data/dataset/CAI-SWTB-Dataset'
    train_dir = os.path.join(data_dir, 'Train')
    validation_dir = os.path.join(data_dir, 'Validation')   
    test_dir = os.path.join(data_dir, 'Test')
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(validation_dir, data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=8, shuffle=True, num_workers=2),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=8, shuffle=True, num_workers=2),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=8, shuffle=True, num_workers=2)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    since = time.time()

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Cuda:', torch.cuda.is_available())
        print('-' * 10)

        #эпоха - обучение и валидация
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  #ОБУЧЕНИЕ
            else:
                model.eval()   #ОЦЕНКА

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []
            for inputs, labels in tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter:"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # оптимизац
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))

            #Ошибка на валидации уменьшилась
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('ВЫПОЛНЕНО ЗА {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('ЛУЧШЕЕ ЗНАЧЕНИЕ: {:4f}'.format(best_val_loss))

    #лучшие веса модели
    model.load_state_dict(best_model_wts)

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    torch.save(model.state_dict(), '/data/best_model.pth')




@app.local_entrypoint()
def main():
    train_model.remote()
