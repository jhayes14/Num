#!/usr/bin/python 
# coding: utf-8

import os,glob,sys
from datetime import datetime
from random import random,randint,choice

import matplotlib.pyplot as plt
import numpy as np

import PIL
from PIL import Image,ImageFilter,ImageChops

#Usage: python ./rotate_offline.py /home/xxx/data/train
#rotation_angles = [180]


def create_image_modifications(image,rotation_angles = [90, 180, 270],translate=True,flip=False,resize=True,shear=True):
    
    if rotation_angles is None:
      imgs = []     
    else:     
      imgs = [rotate_image(image,a) for a in rotation_angles] 
      
    if flip:
	#create mirrors
	imgs.append(flip_image(image,mode='fliplr'))
	imgs.append(flip_image(image,mode='fliptb'))
    
    if translate:
	#random translation
	imgs.append(translate_image(image,'trans1'))
	imgs.append(translate_image(image,'trans2'))
	
    if resize:	
	imgs.append(resize_image(image,mode='resize1',crop=-5))
	imgs.append(resize_image(image,mode='resize2',crop=4))
    
    if shear:
	imgs.append(shearImage(image,mode='shear1',m=-0.3))
	imgs.append(shearImage(image,mode='shear2',m=0.3))
    
    return imgs

def shearImage(image,mode='shear',m=-0.3):
    width, height = image.size
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = image.convert("RGBA").transform((new_width, height), Image.AFFINE,
        (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
    
    fff = Image.new('RGBA', img.size, (255,)*4)
    img = Image.composite(img, fff, img)
    return img,mode
    

def rotate_image(image,angle):
    img = image.convert("RGBA").rotate(angle, expand = 1)
    bg = Image.new("RGB", img.size, PIL.ImageColor.getrgb("white"))
    bg.paste(img, img)
    return bg,angle

def flip_image(image,mode='fliplr'):
    if 'fliplr' in mode:
	fliptype=PIL.Image.FLIP_LEFT_RIGHT
    else:
	fliptype=PIL.Image.FLIP_TOP_BOTTOM
	
    img_mirror = image.convert("RGBA").transpose(fliptype)
    return img_mirror, str(mode)

def translate_image(image,mode='trans'):
    rg = range(-10,-4)+range(5,11)
    tx = choice(rg)
    ty = choice(rg)
    img = ImageChops.offset(image,tx,ty)    

    return img,mode


def resize_image(image,mode='resize',crop=4):
    size = image.size[0],image.size[1]  
    image_resized = image.convert("RGBA").transform(size, Image.EXTENT, (0+crop, 0+crop, image.size[0]-crop, image.size[1]-crop))
    if crop<0:
      fff = Image.new('RGBA', image.size, (255,)*4)
      image_resized = Image.composite(image_resized, fff, image_resized)
    return image_resized,mode

def makeGradient(image):
    img = img.filter(ImageFilter.FIND_EDGES)      
    w, h = img.size
    cs = 1
    img = img.crop((cs, cs, w-cs, h-cs))
    return img,'grad'
  
    
def getDirectoryNames(location="train"):
    directory_names = list(set(glob.glob(os.path.join(location, "*"))).difference(set(glob.glob(os.path.join(location,"*.*")))))
    return sorted(directory_names)

if __name__ == '__main__':  
    translateOnly=False
    if len(sys.argv) < 2:
	raise ValueError('Must pass directory of images as the first parameter')

    img_dir = getDirectoryNames(sys.argv[1])
    if not img_dir:
       img_dir = [sys.argv[1]]
    
    for img_data in img_dir:
	filenames = [ os.path.join(img_data,f) for f in os.listdir(img_data) if os.path.isfile(os.path.join(img_data,f)) ]
	n_images = len(filenames)
	print 'Processing %i images in %s' % (n_images,img_data)
    
	start_time = datetime.now()
    
	for i, imgf in enumerate(filenames):
	    spimgf = imgf.split('/')
	    image_path = '/'.join(spimgf[:-1])
	    image_file = spimgf[-1].split('.')[0]
	    #print "Rd:",random()
	    #print "Open:",imgf
	    img = Image.open(imgf)
	    
	    
	    if translateOnly:
	      img,label = translate_image(img)
	      imgp = image_path + '/' + image_file + '_mod'+label+'.jpg'
	      print imgp
	      #img.save(imgp)
	      
	    else:
	      rimgs = create_image_modifications(img)
	      for rimg, rot in rimgs:
		  rimg.save(image_path + '/' + image_file + '_mod' + str(rot) + '.jpg')
    
	    if ((i+1) % 10000) == 0:
		print 'Processed %i files in %is' % (i+1, (datetime.now() - start_time).seconds)
