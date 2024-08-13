import cv2
import numpy
import os
from glob import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
from torch.autograd import Variable
def functional_conv2d(im):
	sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
	sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
	weight = Variable(torch.from_numpy(sobel_kernel))
	edge_detect = F.conv2d(Variable(im), weight)
	return edge_detect


def rgb2gray(rgb):
	r, g, b = rgb[ :, :, 0], rgb[ :, :,1], rgb[ :, :,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray
class Totensor(object):
	def __call__(self, sample):
		input1=sample['input1']
		input2=sample['input2']
		labels=sample['label']
		at_train=sample['at_train']
		at_train2=sample['at_train2']
		maps=sample['map']
		map2=sample['map2']
		input3=sample['input3']
		inputs1=np.ascontiguousarray(np.transpose(input1,(2,0,1)))
		inputs2=np.ascontiguousarray(np.transpose(input2,(2,0,1)))
		inputs3=np.ascontiguousarray(np.transpose(input3,(2,0,1)))
		at_train=np.ascontiguousarray(np.transpose(at_train,(2,0,1)))
		at_train2=np.ascontiguousarray(np.transpose(at_train2,(2,0,1)))
		labels=np.ascontiguousarray(np.transpose(labels,(2,0,1)))
		maps=np.ascontiguousarray(np.transpose(maps,(2,0,1)))
		map2=np.ascontiguousarray(np.transpose(map2,(2,0,1)))
		return {"input1":torch.from_numpy(inputs1).float()/255.0,
				"input2":torch.from_numpy(inputs2).float()/255.0,
				'input3':torch.from_numpy(inputs3).float()/255.0,
				'at_train':torch.from_numpy(at_train).float()/255.0,
				'at_train2': torch.from_numpy(at_train2).float() / 255.0,
				"labels":torch.from_numpy(labels).float()/255.0,
				"maps":torch.from_numpy(maps).float(),
				'map2':torch.from_numpy(map2).float()}




def gauss_noise(img,sigma=5):
	tem_img=np.float64(np.copy(img))
	h=tem_img.shape[0]
	w=tem_img.shape[1]
	noise=np.random.randn(h,w)*sigma
	noise_image=np.zeros(tem_img.shape,np.float64)
	noise_image[:,:,0]=tem_img[:,:,0]+noise
	noise_image[:,:,1]=tem_img[:,:,1]+noise
	noise_image[:,:,2]=tem_img[:,:,2]+noise
	return noise_image


def get_patch(img1,img2):
	patchsize=128
	x=random.randrange(0,img1.shape[0]-128)
	y=random.randrange(0,img1.shape[1]-128)
	return img1[x:x+patchsize,y:y+patchsize],img2[x:x+patchsize,y:y+patchsize]

class dataset(Dataset):
	def __init__(self,train_hr,transform=None):
		super(dataset,self).__init__()
		self.hr=train_hr
		self.transform=transform
		self.hr_name=[os.path.join(self.hr,i) for i in os.listdir(self.hr)]
	def __len__(self):
		return len(self.hr_name)
	def __getitem__(self, item):
		img_hr=cv2.imread(self.hr_name[item])
		w=int(img_hr.shape[1]/2)
		ilabel=img_hr[:,:w,:]
		imap=img_hr[:,w:,]//255
		ilabel,imap=get_patch(ilabel,imap)
		map=imap[:,:,0:1]
		map2=1-map
		iBlur = cv2.GaussianBlur(ilabel, (7, 7), sigmaX=0, sigmaY=0)
		ia=ilabel*imap+iBlur*(1-imap)
		ib=iBlur*imap+ilabel*(1-imap)
		x=ia
		y=ib
		ia=cv2.resize(ia,(64,64),interpolation=cv2.INTER_CUBIC)
		ib=cv2.resize(ib,(64,64),interpolation=cv2.INTER_CUBIC)
		ic=cv2.resize(ilabel,(64,64),interpolation=cv2.INTER_CUBIC)
		sample={'input1':ia,'input2':ib,'at_train':x,'at_train2':y,'label':ilabel,'map':map,'map2':map2,'input3':ic}
		if self.transform is not None:
			sample=self.transform(sample)
		return sample





def data_loader(opts):
	transforms=Totensor()
	train_set=dataset(opts["MODEL"]["G"]["TRAIN_PATH"],transform=transforms)
	data_loader=torch.utils.data.DataLoader(train_set,batch_size=opts["MODEL"]["G"]["BATCH_SIZE"],shuffle=True,num_workers=opts["MODEL"]["G"]["NUM_WORKERS"])
	return data_loader









