import torch
from torchvision.transforms.functional import to_tensor
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import time
from torch.utils.data import Dataset
from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset

def my_collate_fn(batch):
    imgs, labels, fns = zip(*batch)
    imgs = [to_tensor(img) for img in imgs]#转为张量
    labels = torch.tensor(labels)  # 将标签转换为张量
    return torch.stack(imgs), labels, list(fns)
def main(jihuo):
	# parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parse.add_argument('--name', '-n', type=str, default='fs_xception_c0_299')
	# parse.add_argument('--train_list', '-tl' , type=str, default = 'dfdc_train.txt')
	# parse.add_argument('--val_list', '-vl' , type=str, default = 'dfdc_test.txt')
	# parse.add_argument('--batch_size', '-bz', type=int, default=128)
	# parse.add_argument('--epoches', '-e', type=int, default='10')
	# parse.add_argument('--model_name', '-mn', type=str, default='best.pkl')
	# parse.add_argument('--continue_train', type=bool, default=True)
	# parse.add_argument('--model_path', '-mp', type=str, default='ffpp_c23.pth')
	# args = parse.parse_args()
	name='fs_xception_c0_299'
	jihuo=jihuo
	continue_train =True
	train_list = '/disk/disk2/ghq/path/df_path/dfdc_train.txt'
	val_list = '/disk/disk2/ghq/path/df_path/dfdc_test.txt'
	epoches =50
	batch_size = 64
	model_name = 'best.pkl'
	model_path ='ffpp_c23.pth'
	output_path = os.path.join('output/', name)
	canshu_path=os.path.join(output_path, "best.pkl")
	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)
	torch.backends.cudnn.benchmark=True
	train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
	val_dataset = MyDataset(txt_path=val_list)
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8,collate_fn=my_collate_fn)
 
	train_dataset_size = len(train_dataset)
	val_dataset_size = len(val_dataset)
	model = model_selection(modelname='xception',num_out_classes=2,jihuo=jihuo, dropout=0.5)
	#global sum
	#total_params = sum(p.numel() for p in model.parameters())
	#print(f"总参数量: {total_params}")
	if continue_train:
		# if (os.path.exists(canshu_path)):
		# 	model.load_state_dict(torch.load(canshu_path))
		if model.state_dict():
			print("已加载预训练模型")
		else:
			print("无加载预训练模型")
	if torch.cuda.is_available():
		model = model.cuda()
		print("training on cuda\n")
	criterion = nn.CrossEntropyLoss()#交叉损失函数
	learingrate=0.0001
	print('learingrate: {:.4f} '.format(learingrate))
	#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
	optimizer = optim.Adam(model.parameters(), lr=learingrate, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	model = nn.DataParallel(model)
	best_model_wts = model.state_dict()
	best_acc = 0.0
	iteration = 0
	time_totalstart=time.time()
	train_loss = []
	train_acc = []
	for epoch in range(epoches):
		time_epochstart=time.time()
		print('Epoch {}/{}'.format(epoch+1, epoches))
		print('-'*10)
		model.train()
		train_loss = 0.0
		train_corrects = 0.0
		val_loss = 0.0
		val_corrects = 0.0
		sum=0
		for (image, labels,fn) in train_loader:
			iter_loss = 0.0
			iter_corrects = 0.0
			if torch.cuda.is_available():
				image = image.cuda()
				labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			iter_loss = loss.data.item()
			train_loss += iter_loss
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)#每一次iter正确的数量
			train_corrects +=iter_corrects
			iteration += 1#迭代次数+1
			if not (iteration % 50):#迭代了20次之后
				print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
		epoch_loss = train_loss / train_dataset_size
		epoch_acc = train_corrects / train_dataset_size  
		time_epochend=time.time()-time_epochstart
		print('epoch train loss: {:.4f} Acc: {:.4f} time:{:.4f}'.format(epoch_loss, epoch_acc,time_epochend))

		model.eval()
		dict = {}
		corrects=0
		
		with torch.no_grad():
			for (image, labels,fn) in val_loader:
				if torch.cuda.is_available():
					image = image.cuda()
					labels = labels.cuda()
				outputs = model(image)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				val_loss += loss.data.item()
				#val_corrects += torch.sum(preds == labels.data).to(torch.float32)
				for i in range(batch_size):#对每一个loader操作
					video_name = os.path.basename(fn[i]).split('_')[0]#名字
					label=int(labels[i])#标签
					pread=int(preds[i].data)
					if video_name in dict:#累计判断
						if pread==1:
							dict[video_name]['count'] += 1
						else:
							dict[video_name]['count'] -= 1
					else:#第一次
						if pread==0:
							dict[video_name] = {'label': label, 'count': -1}
						else:
							dict[video_name] = {'label': label, 'count': 1}
			epoch_loss = val_loss / val_dataset_size
			print('epoch val loss: {:.4f}'.format(epoch_loss))
			for i in dict:#对count进行处理
				if dict[i]['count']>=0:
					dict[i]['count']=1
				else:
					dict[i]['count']=0
			for i in dict:
				if dict[i]['count']==dict[i]['label']:
					corrects+=1			
			epoch_acc=corrects/len(dict)
			print('epoch val Acc: {:.4f}'.format(epoch_acc))
			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
		scheduler.step()
		#if not (epoch % 40):
		torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
	time_elapsed=time.time()-time_totalstart
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:.4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	torch.save(model.module.state_dict(), canshu_path)#一条语句保存参数
if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = "2"  #（代表仅使用第0，1号GPU）
	for i in range(3):
		if i == 0:
			jihuo = nn.ReLU(inplace=True)
			print('relu')
		elif i == 1:
			jihuo = nn.ReLU6(inplace=True)
			print('relu6')
		elif i == 2:
			jihuo = nn.Sigmoid()
			print('sigmoid')
		main(jihuo) 		
#CUDA_VISIBLE_DEVICES=2,3 python train_CNN.py 
#nohup python train.py >dfdc_3jihuo_3.txt 2>&1 &
#kill  -15  pid号
#/disk/disk1/conda/envs/ymj/bin/python