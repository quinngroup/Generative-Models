import cv2
import numpy as np
import argparse

'''
Converts numpy arrays to videos

@author Meekail Zain
'''
parser = argparse.ArgumentParser(description='array2vid')
parser.add_argument('--save', type=str, default='', metavar='s',help='saves video under this name')
parser.add_argument('--load', type=str, default='', metavar='l',help='loads the given numpy array')
args = parser.parse_args()

#Determine FPS and respective ms delay
fps=10
delay=1000//fps


#Loads raw numpy array (hence there are size-limits) and establishes video dimensions
assert args.load!='','Please specify array to convert'
raw=np.load(args.load,allow_pickle=True)
#raw=raw[:,0,:,:]
print(raw.shape)
frame_shape=raw[0].shape
length=raw.shape[0]
raw=np.uint8(raw)



#Construct video-writer using given save name
if(args.save!=''):
    name=args.save
else:
    name=args.load[:-4]

#Works on moving Mnist with encoding (MP42 or -1) and .mp4 in spite of error
#encoding=cv2.VideoWriter_fourcc(*'mp4v')
#encoding=cv2.VideoWriter_fourcc(*'MP42')
encoding=cv2.VideoWriter_fourcc(*'BID ')
out = cv2.VideoWriter(name+'.mp4',encoding, fps, frame_shape)
temp=np.zeros_like(raw[0])
for i in range(length):
    print(i)    
    temp=cv2.cvtColor(raw[i,:,:],cv2.COLOR_GRAY2BGR)
    #out.write(temp)
    #temp=np.uint8(np.random.rand(frame_shape[0],frame_shape[1])*255)
    out.write(temp)
    cv2.imshow('frame',temp)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break


#Close
out.release()

cv2.destroyAllWindows()
