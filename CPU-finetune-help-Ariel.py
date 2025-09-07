#%% [markdown]
# ### Ayuda para correr el script
#%%
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional,Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import Dataset
from transformers import ViTModel, ViTConfig
from transformers import modeling_outputs
import torchvision.transforms.functional as F
from sklearn.model_selection import StratifiedKFold
import tifffile as tiff

#%%

EPOCHS = 180
checkpoint_path = 'NONE'
LR = 1e-6
# NUM_WORKERS = 8

PATH = Path(__file__).parent.resolve()
images_path = PATH / 'dataset'
labels_path = PATH / 'patch_labels.csv'

# torch.set_num_threads(NUM_WORKERS)

def today():
    return time.strftime("%d_%m_%Y")

# %%
def blur_kernel(size, channels = 1):
        kernel = torch.zeros((size, size), dtype=torch.float32)
        kernel[size // 2, :] = torch.ones(size)
        kernel = kernel / size
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        
        
        return kernel

k = blur_kernel(size=3, channels=3)


class MotionBlur:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        
        padding = self.kernel_size //2
        
        
        img = img.unsqueeze(0)  
        channels = img.shape[1]  
        
        
        
        blurred = nn.functional.conv2d(img, k, padding=padding, groups=channels)
        blurred = blurred.squeeze(0)
    
        
        
        return blurred


# %%

def custom_collate(batch):
    
    pixel_values = torch.stack([example['image'].clone().detach() for example in batch])
    labels = torch.tensor([example['labels'] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def train_transforms_3channel(examples_):
    size = 224
    
    t_choice = np.random.choice(np.arange(5), p=[0.2, 0.2, 0.1, 0.1, 0.4])

    
    augment_transform = {
        0: transforms.RandomHorizontalFlip(),
        1: transforms.RandomVerticalFlip(),
        2: transforms.RandomResizedCrop(size, scale=(0.85, 1)),
        3: MotionBlur(kernel_size=3),
        4: transforms.Lambda(lambda x: x)
    }.get(t_choice)
    
    
    transforms_ = transforms.Compose([
        transforms.Lambda(lambda x: 400 + np.clip(x, -1100, 300)),
        transforms.Lambda(lambda x: x /700 ),
        transforms.Lambda(lambda x: x.astype(np.float32)),
        transforms.ToTensor(),
        augment_transform,
    ])
    
    examples_['image'] = [transforms_(image) for image in examples_['image']]

    
    return examples_

def test_transforms_3channel(examples_):

    
    transforms_ = transforms.Compose([
        transforms.Lambda(lambda x: 400 + np.clip(x, -1100, 300)),
        transforms.Lambda(lambda x: x /700 ),
        transforms.Lambda(lambda x: x.astype(np.float32)),
        transforms.ToTensor(),        
        ])

    examples_['image'] = [transforms_(image) for image in examples_['image']]

    return examples_

        
images = sorted(list(images_path.glob('*.tiff')), key=lambda x: int(x.stem.split('resized_patch_')[-1] ))
labels = pd.read_csv(labels_path).values.flatten().tolist()

dataset_dict = {'image': [], 'labels': []}
label2name = {"1": "0NLP", "2": "1CLE", "3": "2PSE"}




for i, _ in enumerate(labels):
    dataset_dict['image'].append(np.array(tiff.imread(str(images[i]))))
    dataset_dict['labels'].append(label2name[str(labels[i])])

ds = Dataset.from_dict(dataset_dict)
ds = ds.class_encode_column('labels')


splits4test = ds.train_test_split(test_size=0.25, stratify_by_column='labels', seed=42)
train_ds = splits4test['train']
test_ds = splits4test['test']           #El test set queda fijo

test_ds.set_transform(test_transforms_3channel)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=custom_collate)
# test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=custom_collate,num_workers=NUM_WORKERS)

kf = StratifiedKFold(n_splits=5)

train_indices = []
val_indices = []

for t_idx, v_idx in kf.split(train_ds, train_ds['labels']):
    train_indices.append(t_idx)
    val_indices.append(v_idx)
    
train_ds = []
train_loaders = []
val_ds = []
val_loaders = []


versions = [f'{today()}_v{k_fold}' for k_fold in range(5)]

for k_fold, _ in enumerate (versions):
    train_ds = splits4test['train'].select(train_indices[k_fold])
    train_ds.set_transform(train_transforms_3channel)
    train_loaders.append(DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=custom_collate))
    # train_loaders.append(DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=custom_collate,num_workers=NUM_WORKERS))
    val_ds = splits4test['train'].select(val_indices[k_fold])
    val_ds.set_transform(test_transforms_3channel)
    val_loaders.append(DataLoader(val_ds, batch_size=2, shuffle=True, collate_fn=custom_collate))
    # train_loaders.append(DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=custom_collate,num_workers=NUM_WORKERS))
    


# %%
# dataiter = iter(train_loader)
# data = next(dataiter)


# images = data['pixel_values']

# fig, axes = plt.subplots(2, 2, figsize=(6, 6))

# titles = [label2name[str(data['labels'][j].item()+1)] for j in range(len(data['labels']))]

# multi_channel = []
# for idx, ax in enumerate(axes.flatten()):
#     img = images[idx]
#     img = (1+img.permute(1,2,0).numpy())/2
    
#     ax.imshow(img, cmap='gray')
#     ax.set_title(titles[idx])
#     ax.axis('off')
# # plt.savefig(PATH / 'train_transforms.png', dpi=300, bbox_inches='tight')
# plt.tight_layout()

# # plt.savefig(r'C:\Users\lzuni\Documents\Tesis\resultados\train_transforms.png', dpi=300, bbox_inches='tight')
# plt.show()

# %%
# plt.savefig(PATH / 'train_transforms.png', dpi=300, bbox_inches='tight')
# plt.savefig(r'C:\Users\lzuni\Documents\Tesis\resultados\train_transforms.png', dpi=300, bbox_inches='tight')


# %%

class Net(nn.Module):
    def __init__(self, model_name, config: ViTConfig):
        super(Net, self).__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=True)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, modeling_outputs.ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        
        sequence_output = outputs['last_hidden_state']
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None

        if not return_dict:
            return logits

        return modeling_outputs.ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# %% [markdown]
# ### Initialize model

# %%
model_name = "google/vit-base-patch32-224-in21k"
id2label = {"0": "0NLP", "1": "1CLE", "2": "2PSE"}
label2id = {label:id for id,label in id2label.items()}

config = ViTConfig.from_pretrained(model_name)
config.label2id = label2id
config.id2label = id2label

net = Net(model_name, config)
# net = net.cuda()
for name, param in net.named_parameters():
        # if 'classifier' in name:
        if 'embeddings' in name or 'classifier' in name or 'pooler' in name:
        # if 'embeddings.cls_token' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=LR,
    weight_decay=1e-5,
    eps=1e-09,
    amsgrad=True,
    betas= (0.9, 0.98)
    )

evolution_paths = [PATH / f'results_{ver}.csv' for ver in versions]

net_crossval = [net for k in range(len(versions))]
optim_crossval = [optimizer for k in range(len(versions))]

if checkpoint_path == 'NONE':
    print('Training from scratch!')
    start_epoch = 0
    best_epoch = 0
    evolution = pd.DataFrame({'Epoch': [], 
                            'Train loss': [], 
                            'Train accuracy': [], 
                            'Validation loss': [], 
                            'Validation accuracy': []})
    [evolution.to_csv(evolution_path, mode='w', header=True, index=False, sep=' ') for evolution_path in evolution_paths]


else:
    print('Loading model to continue training!')
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    net.load_state_dict(checkpoint['trainable_state_dict'], strict=False)
    start_epoch = checkpoint['saved_epoch'] + 1
    best_epoch = checkpoint['saved_epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    evolution = pd.DataFrame({'Epoch': [], 
                            'Train loss': [], 
                            'Train accuracy': [], 
                            'Validation loss': [], 
                            'Validation accuracy': []})
    [evolution.to_csv(evolution_path, mode='a', header=False, index=False, sep=' ') for evolution_path in evolution_paths]



# print(np.array( [name for name, param in net.named_parameters() if param.requires_grad]))


# %%
#Celda opcional para guardar la cantidad de parámetros y la configuración del modelo

total_params = sum(p.numel() for p in net_crossval[4].parameters())
print(f"Total number of parameters in the network: {total_params}")


trainable_params = sum(p.numel() for p in net_crossval[4].parameters() if p.requires_grad)
print(f"Number of trainable parameters in the network: {trainable_params}")

# with open(PATH / f'config_{ver}.txt', 'w') as f:
#     f.write(f"model_name: {model_name}\n")
#     f.write(f"Total number of parameters in the network: {total_params}")
#     f.write(f"Number of trainable parameters in the network: {trainable_params}\n")
#     for key, value in config.to_dict().items():
#         f.write(f"{key}: {value}\n")


# %% [markdown]
# ### Celda opcional para retomar entrenamiento desde alguna época

# %%
# net = Net(config)

            
# checkpoint = torch.load(PATH / f'last_model_fine_{ver}.pt', weights_only=True)
# net.load_state_dict(checkpoint['trainable_state_dict'], strict=False)

# optimizer = torch.optim.AdamW(
#     net.parameters(),
#     lr=1e-6,
#     weight_decay=1e-5,
#     eps=1e-09,
#     amsgrad=True,
#     betas= (0.9, 0.98)
#     )
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# for name, param in net.named_parameters():
        
#         if 'embeddings' in name or 'classifier' in name or 'pooler' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

# start_epoch = checkpoint['saved_epoch'] +1
# evolution = pd.DataFrame({'Epoch': [], 
#                               'Train loss': [], 
#                               'Train accuracy': [], 
#                               'Validation loss': [], 
#                               'Validation accuracy': []})
# evolution.to_csv(evolution_path, mode='a', header=False, index=False, sep=' ')


# %%
def one_epoch(net, optimizer, criterion, loader, group):
    if group == 'train':
        net.train()
    else:
        net.eval()
    running_loss = 0.
    avg_loss = 0.
    tp = 0

    for i, data in enumerate(loader):
        
        inputs = data['pixel_values']        
        labels = data['labels']

        # inputs = inputs.cuda()
        # labels = labels.cuda()
        if group == 'train':
            optimizer.zero_grad()
            outputs = net(inputs).logits
        else:
            with torch.no_grad():
                outputs = net(inputs).logits
        loss = criterion(outputs, labels)
        outputs_soft = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs_soft, dim=1)
        tp += (preds == labels).sum().item()

        if group == 'train':
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        

    avg_loss = running_loss / (i+1) # loss per epoch
    avg_acc = tp / len(loader.dataset) # accuracy per epoch

    return avg_acc, avg_loss


#%% [markdown]
# ### Entrenamiento
#%%

best_vloss = 1_000_000.


for k_fold, ver in enumerate(versions):
    print('Training fold ', k_fold)
    start_time = time.time()
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch+EPOCHS):
    
    
        if epoch % 5 == 0:
            print(f'epoch: {epoch}', f'best_epoch: {best_epoch}') 

        avg_acc, avg_loss = one_epoch(net_crossval[k_fold], optim_crossval[k_fold], criterion, train_loaders[k_fold], 'train')
        avg_vacc, avg_vloss = one_epoch(net_crossval[k_fold], optim_crossval[k_fold], criterion, val_loaders[k_fold], 'val')

        evolution = pd.DataFrame({'epoch': [epoch+1], 
                                'Train loss': [round(avg_loss, 4)], 
                                'Train accuracy': [round(avg_acc, 4)], 
                                'Validation loss': [round(avg_vloss, 4)], 
                                'Validation accuracy': [round(avg_vacc, 4)]})
        
        evolution.to_csv(evolution_paths[k_fold], mode='a', header=False, index=False, sep=' ')


        # Save the best model

        if avg_vloss < best_vloss:
            
            current_loss = avg_vloss
            best_vloss = avg_vloss
            best_epoch = epoch

            trainable_state_dict = {
                name: param.detach().cpu()
                for name, param in net_crossval[k_fold].named_parameters()
                if 'embeddings' in name or 'classifier' in name or 'pooler' in name
            }

            checkpoint = {
                'trainable_state_dict': trainable_state_dict,
                'optimizer_state_dict': optim_crossval[k_fold].state_dict(),
                'saved_epoch': best_epoch,
                'train_loss': current_loss,
                'val_loss': best_vloss,
            }
        

    if epoch - best_epoch > 5:
        torch.save(checkpoint,  PATH / f'best_model_fine_{ver}.pt')

    trainable_state_dict2 = {
            name: param.detach().cpu()
            for name, param in net_crossval[k_fold].named_parameters()
            if 'embeddings' in name or 'classifier' in name or 'pooler' in name
        }
        

    checkpoint2 = {
                'trainable_state_dict': trainable_state_dict2,
                'optimizer_state_dict': optim_crossval[k_fold].state_dict(),
                'saved_epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': avg_vloss,
            }



    torch.save(checkpoint2,  PATH / f'last_model_fine_{ver}.pt')

    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')



# %%
# # evolution_path = PATH / f'{ver}.csv'
evolution = pd.read_csv(evolution_paths[0], sep=' ')

divs = 10
fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].set_ylim(0, np.max(evolution['Train loss']) + 0.1)
axes[0].set_xticks(np.arange(0, evolution['Epoch'].max() + 1, divs))

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].plot(evolution['Epoch'], evolution['Train loss'], c='b', label='Training Loss')
axes[0].plot(evolution['Epoch'], evolution['Validation loss'], c='r', label='Validation Loss')
axes[0].grid()
axes[0].legend()

axes[1].set_ylim(0, 1)
axes[1].set_xticks(np.arange(0, evolution['Epoch'].max() + 1, divs))

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].plot(evolution['Epoch'], evolution['Train accuracy'], c='b', label='Training Accuracy')
axes[1].plot(evolution['Epoch'], evolution['Validation accuracy'], c='r', label='Validation Accuracy')
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.savefig(PATH / f'evolution_plot_v{ver}.png', dpi=300, bbox_inches='tight')



# %%
