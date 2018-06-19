import numpy as np

clean = [711,2558,2604,3489,3582,4071, 4178,4226,4308,5059,6517, 7193,7396,7483,7789]

# get files
f = open('unpadded_val_txt_files/image_files.txt', 'rb')
image_files = f.readlines()
f.close()

f = open('unpadded_val_txt_files/word_files.txt', 'rb')
word_files = f.readlines()
f.close()

f = open('unpadded_val_txt_files/y_labels.txt', 'rb')
y_labels = f.readlines()
f.close()

print len(image_files)
for i in sorted(clean, reverse=True):
	del image_files[i]
	del word_files[i]
	del y_labels[i]

print 'saving labels'
f = open('unpadded_val_txt_files/y_labels.txt', 'w')
for item in y_labels:
  f.write("%s" % item)
f.close()

print 'saving image names'
f = open('unpadded_val_txt_files/image_files.txt', 'w')
for item in image_files:
  f.write("%s" % item)
f.close()

print 'saving word names'
f = open('unpadded_val_txt_files/word_files.txt', 'w')
for item in word_files:
  f.write("%s" % item)
f.close()

