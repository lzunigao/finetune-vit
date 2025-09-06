import argparse
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--EPOCHS', type=int, default=50,
                        help='Number of epoch (default: 50)')
    parser.add_argument('--K_FOLD', type=int, default=0,
                        help='Number of epoch (default: 0)')
    parser.add_argument('--CHECKPOINT_PATH', type=str, default='NONE',
                        help='Path to the model to load and continue training (default: None)')
    parser.add_argument('--LR', type=float, default=1e-6,
                        help='Learning rate (default: 1e-6)')
    return vars(parser.parse_args())
    



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

# %%
class Net(nn.Module):
    def __init__(self, config: ViTConfig):
        super(Net, self).__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.vit = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k', add_pooling_layer=True)

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
        # sequence_output = outputs[0]
        sequence_output = outputs['last_hidden_state']

        
        logits = self.classifier(sequence_output[:, 0, :])
        # aux = self.pre_classifier(sequence_output)
        # aux = F.relu(aux)             
        loss_fct = nn.CrossEntropyLoss( )
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss = None
        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) 
            return ((loss,) + output) if loss is not None else output

        return modeling_outputs.ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    kwargs = parse_args()

    if kwargs['CHECKPOINT_PATH']=='NONE':
        checkpoint_path='NONE'
        print('Checkpoint is NONE, training from scratch!')

    else:
        checkpoint_path = Path(kwargs['CHECKPOINT_PATH'])
        if checkpoint_path.exists():
            print('Loading checkpoint from:', checkpoint_path)
        else:
            print('Failure to load checkpoint')
            sys.exit()

    EPOCHS = kwargs['EPOCHS']
    k_fold = kwargs['K_FOLD']
    LR = kwargs['LR']
    PATH = Path(r'/home/laura.zuniga/finetune_net')
    if not PATH.exists():
        PATH.mkdir(parents=True, exist_ok=True)
    images_path = Path(r'/home/laura.zuniga/datasets/resized_patches_224px')
    labels_path = Path(r'/home/laura.zuniga/datasets/EmphysemaCT_sorense')
    # model_path = Path(r'/home/laura.zuniga/downloaded_nets')

# # import torchvision
# # import torch.nn.functional as F
# # from torch.utils.tensorboard import SummaryWriter
# # from PIL import Image as PILImage


# # from transformers import ViTForImageClassification

# # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# # from torchvision.utils import make_grid

# # import cv2
# # import random

# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import LabelBinarizer
# # from sklearn.metrics import RocCurveDisplay, auc, roc_curve
# # from itertools import cycle

    # %%
    # using vsc
    def today():
        return time.strftime("%d_Sep_%Y",)

            



    NUM_WORKERS = 8
    torch.set_num_threads(NUM_WORKERS)

        








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
        
        t_choice = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.1, 0.1, 0.4])

        
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
    labels = pd.read_csv(labels_path / 'patch_labels.csv').values.flatten().tolist()

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


    kf = StratifiedKFold(n_splits=5)

    #k_fold = 0

    train_indices = []
    val_indices = []

    for t_idx, v_idx in kf.split(train_ds, train_ds['labels']):
        train_indices.append(t_idx)
        val_indices.append(v_idx)
        
        
    train_ds = splits4test['train'].select(train_indices[k_fold])
    val_ds = splits4test['train'].select(val_indices[k_fold])


    train_ds.set_transform(train_transforms_3channel)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=custom_collate,num_workers=NUM_WORKERS)


    test_ds.set_transform(test_transforms_3channel)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=custom_collate,num_workers=NUM_WORKERS)


    val_ds.set_transform(test_transforms_3channel)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=True, collate_fn=custom_collate,num_workers=NUM_WORKERS)


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


        




    # %% [markdown]
    # ### Initialize model

    # %%

    ver = f'{today()}_v{k_fold}' # update version
    # print('version', ver)

    evolution_path = PATH / f'{ver}.csv'

    # model_name = "google/vit-base-patch16-224"
    id2label = {"0": "0NLP", "1": "1CLE", "2": "2PSE"}
    label2id = {label:id for id,label in id2label.items()}

    config = ViTConfig.from_pretrained("google/vit-base-patch32-224-in21k")

    config.label2id = label2id
    config.id2label = id2label




    net = Net(config)
    net = net.cuda()



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LR,
        weight_decay=1e-5,
        eps=1e-09,
        amsgrad=True,
        betas= (0.9, 0.98)
        )



    if checkpoint_path == 'NONE':
        print('Training from scratch!')
        start_epoch = 0
        best_epoch = 0
        evolution = pd.DataFrame({'Epoch': [], 
                                'Train loss': [], 
                                'Train accuracy': [], 
                                'Validation loss': [], 
                                'Validation accuracy': []})
        evolution.to_csv(evolution_path, mode='w', header=True, index=False, sep=' ')


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
        evolution.to_csv(evolution_path, mode='a', header=False, index=False, sep=' ')

    for name, param in net.named_parameters():
            # if 'classifier' in name:
            if 'embeddings' in name or 'classifier' in name or 'pooler' in name:
            # if 'embeddings.cls_token' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # print(np.array( [name for name, param in net.named_parameters() if param.requires_grad]))


    # %%
    #Celda opcional para guardar la cantidad de parámetros y la configuración del modelo

    # total_params = sum(p.numel() for p in net.parameters())
    # print(f"Total number of parameters in the network: {total_params}")


    # trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters in the network: {trainable_params}")

    # with open(PATH / f'config_{ver}.txt', 'w') as f:
    #     f.write(f"model_name: {model_name}\n")
    #     f.write(f"Total number of parameters in the network: {total_params}")
    #     f.write(f"Number of trainable parameters in the network: {trainable_params}\n")
    #     for key, value in config.to_dict().items():
    #         f.write(f"{key}: {value}\n")


    # %% [markdown]
    # ### Load to resume training

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


    # def train_one_epoch(epoch_index, tb_writer):
    def train_one_epoch(epoch_index, net, optimizer, criterion, train_loader):
        running_loss = 0.
        last_loss = 0.

        train_tp = 0
        
        for i, data in enumerate(train_loader):
            
            
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['pixel_values']        
            labels = data['labels']

            inputs = inputs.cuda()
            labels = labels.cuda()


            # zero the parameter gradients
            optimizer.zero_grad()
            
            
            # outputs is a BaseModelOutputWithPooling

            optimizer.zero_grad()
            outputs = net(inputs).logits
            loss = criterion(outputs, labels)

            outputs_soft = torch.softmax(outputs, dim=1)

            

            train_preds = torch.argmax(outputs_soft, dim=1)
            train_tp += (train_preds == labels).sum().item()
            
            
            
            
            # outputs = torch.softmax(outputs, dim=1)
            loss.backward()
            optimizer.step()


            # Gather data and report
            running_loss += loss.item()
            
            
            
        # counter = (len(train_loader.dataset) - len(train_loader.dataset)%len(data['pixel_values']))// len(data['pixel_values'])
        
        last_loss = running_loss / (i+1) # loss per epoch
        
        train_accuracy = train_tp / len(train_loader.dataset) # accuracy per epoch

        return train_accuracy, last_loss

    # %%



    #EPOCHS = 1
    best_vloss = 1_000_000.

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch+EPOCHS):
        
        
        # print(f'epoch: {epoch}', f'best_epoch: {best_epoch}')  if epoch % 5 == 0 else None
        print(f'epoch: {epoch}', f'best_epoch: {best_epoch}') 

        net.train()
        train_accuracy, avg_loss = train_one_epoch(epoch+1, net, optimizer, criterion, train_loader)
        
        running_vloss = 0.0

        net.eval()
        
        with torch.no_grad():
            val_tp = 0

            for i, vdata in enumerate(val_loader):
            
                vinputs = vdata['pixel_values']
                vlabels = vdata['labels']

                vinputs = vinputs.cuda()
                vlabels = vlabels.cuda()

                
                voutputs = net(vinputs).logits

                vloss = criterion(voutputs, vlabels)

                voutputs_soft = torch.softmax(voutputs, dim=1)
                vpreds = torch.argmax(voutputs_soft, dim=1)
                val_tp += (vpreds == vlabels).float().sum()
                
                
                running_vloss += vloss
                
                

        avg_vloss = running_vloss / (i+1)
        
        val_accuracy = val_tp / len(val_loader.dataset) # accuracy per epoch

    
        evolution = pd.DataFrame({'epoch': [epoch+1], 
                                'Train loss': [round(avg_loss, 4)], 
                                'Train accuracy': [round(train_accuracy, 4)], 
                                'Validation loss': [round(avg_vloss.item(), 4)], 
                                'Validation accuracy': [round(val_accuracy.item(), 4)]})
        
        
        evolution.to_csv(evolution_path, mode='a', header=False, index=False, sep=' ')


        # Save the best model

        if avg_vloss < best_vloss:
            
            current_loss = avg_vloss
            best_vloss = avg_vloss
            best_epoch = epoch

            trainable_state_dict = {
                name: param.detach().cpu()
                for name, param in net.named_parameters()
                if 'embeddings' in name or 'classifier' in name or 'pooler' in name
            }

            checkpoint = {
                'trainable_state_dict': trainable_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'saved_epoch': best_epoch,
                'train_loss': current_loss,
                'val_loss': best_vloss,
            }
            


        # if epoch - best_epoch > 5:        
            
            torch.save(checkpoint,  PATH / f'best_model_fine_{ver}.pt')

        trainable_state_dict2 = {
                name: param.detach().cpu()
                for name, param in net.named_parameters()
                if 'embeddings' in name or 'classifier' in name or 'pooler' in name
            }
                
        
    checkpoint2 = {
                'trainable_state_dict': trainable_state_dict2,
                'optimizer_state_dict': optimizer.state_dict(),
                'saved_epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': avg_vloss,
            }



    torch.save(checkpoint2,  PATH / f'last_model_fine_{ver}.pt')
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')
    # print('Finished training')
    # print('Best epoch: {}'.format(best_epoch))


    # # %%
    # # evolution_path = PATH / f'{ver}.csv'
    evolution = pd.read_csv(evolution_path, sep=' ')

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

    # # plt.show()


    # # %%

    # # net = Net(config)
    # # net.add_module('classifier', nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity())

    # # for name, param in net.named_parameters():
    # #     param.requires_grad = False

    # # #test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # # checkpoint = torch.load(PATH / f'last_model_fine_{ver}.pt')

    # # net.load_state_dict(checkpoint['trainable_state_dict'], strict=False)

    # # # optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
    # # criterion = nn.CrossEntropyLoss()
    # # # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # net.eval()

    # # # print saved epoch
    # print('Saved epoch: {}'.format(checkpoint['saved_epoch']))

    # correct = 0
    # total = 0
    # test_loader = DataLoader(test_ds, collate_fn=custom_collate, batch_size=1, shuffle=False)
    # X_test = np.vstack([example['pixel_values'].view(example['pixel_values'].size(0), -1).numpy() for example in test_loader])
    # y_test = np.hstack([example['labels'].numpy() for example in test_loader])

    # predicted_arr = []
    # correct_arr = []
    # loss_test = []


    # with torch.no_grad():
    #     for data in test_loader:
            
    #         images = data['pixel_values']

    #         labels = data['labels']
                
    #         outputs = net(images).logits
            
            
    #         loss_test.append(criterion(outputs, labels))
            
    #         outputs_soft = torch.softmax(outputs, dim=1)

    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs_soft, 1)

            
    #         predicted_arr.append(predicted.item())
    #         correct_arr.append(labels)
            
            
    #         correct += (predicted == labels).sum().item()



    # print(f'Accuracy of the network on the test images: {100 * correct // len(test_loader.dataset)} %')

    # plt.figure(3)
    # cm = confusion_matrix(correct_arr, predicted_arr)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label2id)
    # disp.plot(cmap="Blues", colorbar=False)
    # plt.title('Matriz de confusión')
    # plt.savefig(PATH / f'confusion_finetune_{ver}.png', dpi=300, bbox_inches='tight')
    # # plt.show()

    # plt.figure(4)
    # plt.title('Test Loss histogram')
    # plt.hist(loss_test, bins=20, color='blue', alpha=0.7)
    # plt.xlabel('Loss')
    # plt.ylabel('Frequency')
    # # plt.savefig(PATH / 'hist_fine1.png', dpi=300, bbox_inches='tight')
    # # plt.show()



    # # %%

    # TP = np.array([0, 0, 0])
    # TN = np.array([0, 0, 0])
    # FP = np.array([0, 0, 0])
    # FN = np.array([0, 0, 0])

    # for i in [0, 1, 2]:
    #     TP[i] = cm[i][i]
    #     FP[i] = sum(cm[j][i] for j in range(3) if j != i)
    #     FN[i] = sum(cm[i][j] for j in range(3) if j != i)
    #     TN[i] = sum(cm[j][j] for j in range(3) if j != i)

    # print('TP:', TP)
    # print('FP:', FP)
    # print('FN:', FN)
    # print('TN:', TN)

    # Precision = (TP / (TP + FP)).round(2)
    # Recall = (TP / (TP + FN)).round(2)
    # Specificity = (TN / (TN + FP)).round(2)
    # J = (Recall + Specificity - 1).round(2)
    # kappa = (2*( (TP*TN)-(FN*FP) )/ ( (TP+FP)*(FP+TN)+(TP+FN)*(TN+FN) )).round(2)

    # print('Precision:', Precision)
    # print('Recall:', Recall)
    # print('Specificity:', Specificity)
    # print('J:', J)
    # print('Kappa:', kappa)

    # #save metrics to txt in file named f'metrics_{ver}.txt'
    # with open(PATH / f'metrics_{ver}.txt', 'w') as f:
    #     f.write(f'Accuracy: {100 * correct // len(test_loader.dataset)} %\n')
    #     f.write('Classes: [0 1 2]\n')
    #     f.write(f'Precision: {Precision}\n')
    #     f.write(f'Recall: {Recall}\n')
    #     f.write(f'Specificity: {Specificity}\n')
    #     f.write(f'J: {J}\n')
    #     f.write(f'Kappa: {kappa}\n')

    # # %%
    # X_train = []
    # y_train = []
    # train_loader2 = DataLoader(train_ds, collate_fn=custom_collate, batch_size=1, shuffle=True)
    # for data in train_loader2:
    #     X_train.append(data['pixel_values'].view(data['pixel_values'].size(0), -1).numpy())
    #     y_train.append(data['labels'].numpy())

    # X_train = np.vstack(X_train)
    # y_train = np.hstack(y_train)


    # classifier = LogisticRegression(max_iter=500)

    # y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    # # y_score = classifier.fit(X_train, y_train).predict_proba(X_train)



    # label_binarizer = LabelBinarizer().fit(y_train)
    # y_onehot_test = label_binarizer.transform(y_test)
    # # y_onehot_test = label_binarizer.transform(y_train)
    # n_classes = len(np.unique(y_train))

    # fpr, tpr, roc_auc = dict(), dict(), dict()
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # fpr_grid = np.linspace(0.0, 1.0, 1000)

    # # Interpolate all ROC curves at these points
    # mean_tpr = np.zeros_like(fpr_grid)

    # for i in range(n_classes):
    #     mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # # Average it and compute AUC
    # mean_tpr /= n_classes

    # fpr["macro"] = fpr_grid
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])




    # colors = cycle(["purple", "lime", "gold"])

    # target_names = ["NLP", "CLE", "PSE"]

    # fig, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label=f"Curva ROC micro-avg (AUC = {roc_auc['micro']:.2f})",
    #     color="red",
    #     linestyle=":",
    #     linewidth=4,
    # )

    # plt.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label=f"Curva ROC macro-avg (AUC = {roc_auc['macro']:.2f})",
    #     color="aqua",
    #     linestyle=":",
    #     linewidth=4,
    # )
    # roc = [0,1,2]
    # i = 0
    # for class_id, color in zip(range(n_classes), colors):
    #     roc[i] = RocCurveDisplay.from_predictions(
    #         y_onehot_test[:, class_id],
    #         y_score[:, class_id],
    #         name=f"Curva ROC de {target_names[class_id]}",
    #         color=color,
    #         ax=ax,
    #         plot_chance_level=(class_id == 2),
    #         despine=True,
    #     )

    #     i+= 1
        
    # _ = ax.set(
    #     xlabel="False Positive Rate",
    #     ylabel="True Positive Rate",
    #     title="Extensión de curva ROC\na One-vs-Rest multiclass",
    # )

    # # store the fpr, tpr, and roc_auc for all averaging strategies


    # # save plot in hd
    # plt.savefig(PATH / f'roc_curve_finetune_{ver}.png', dpi=300, bbox_inches='tight')

    # # %% [markdown]
    # # # COLORMAP

    # # %%
    # test_loader2 = DataLoader(
    #     [data for data in test_loader.dataset if data['labels'] == 2],
    #     batch_size=1,
    #     shuffle=True,
    #     collate_fn=custom_collate
    # )


    # with torch.no_grad():
    #     i =0
    #     for data in test_loader2:
    #         if i>0:
    #             break
    #         i+=1

    #         images = data['pixel_values']
    #         img_plt = images[0].permute(1, 2, 0).detach().numpy() + 1 # CHW -> HWC
    #         img_plt = img_plt / 2 # Normalize to [0, 1]

    #         outputs = net(images, output_attentions=True, return_dict=True)
    #         attentions = outputs.attentions
    #         attentions = torch.stack(attentions).squeeze(1)  # shape [batch_size, num_heads, seq_len, seq_len]
    #         att_mean = torch.mean(attentions, dim=1) #average over heads
            
    #         print('att_mean', att_mean.shape)
            
    #         joint_attentions = torch.zeros(att_mean.size())
    #         joint_attentions[0] = att_mean[0]

    #         for n in range(1, att_mean.size(0)):
    #             joint_attentions[n] = torch.matmul(att_mean[n], joint_attentions[n-1])
    #         num_patches = joint_attentions[-1].shape[-1] - 1  # exclude CLS token
    #         side = int(np.sqrt(num_patches)) # e.g., 4 for 16 patches

    #         # Get attention from CLS token to all patches
    #         cls_attn = joint_attentions[-1][0, 1:]  # shape: (num_patches,)

    #         # Reshape to grid
    #         attn_map = cls_attn.reshape(side, side).detach().cpu()

    #         # # Upsample to 64x64.
    #         attn_map = torch.reshape(attn_map, (1, 1, side, side))
    #         attn_map = torch.nn.functional.interpolate(attn_map, size=(64, 64), mode='bilinear', align_corners=False)
    #         attn_map = attn_map[0, 0].detach().cpu().numpy()
            
            
    #         # Plot
    #         fig, ax = plt.subplots( 1,2,figsize=(10, 5))
    #         ax[0].imshow(img_plt, cmap='gray')
    #         ax[0].axis('off')
    #         ax[0].set_title('Input Image')
    #         ax[1].imshow(attn_map, cmap='jet')
    #         ax[1].axis('off')
    #         ax[1].set_title('CLS Attention Map (last layer)')
    #         plt.colorbar(ax[1].imshow(attn_map, cmap='jet'), ax=ax[1], fraction=0.046, pad=0.04)
            
        
    #         plt.savefig(PATH / f'attention_map_{ver}.png', dpi=300, bbox_inches='tight')
    #         plt.show()

    # # %%


    # # %%



