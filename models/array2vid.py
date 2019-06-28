import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='array2vid')
parser.add_argument('--save', type=str, default='', metavar='s',help='saves video under this name')
parser.add_argument('--load', type=str, default='', metavar='l',help='loads the given numpy array')
args = parser.parse_args()

#Determine FPS and respective ms delay
fps=10
delay=int(1000/fps)


#Loads raw numpy array (hence there are size-limits) and establishes video dimensions
assert args.load!='','Please specify array to convert'
raw=np.load(args.load,allow_pickle=True)
frame_shape=raw[0].shape
length=raw.shape[0]
raw=np.uint8(raw)



#Construct video-writer using given save name
if(args.save!=''):
    name=args.save
else:
    name=args.load

out = cv2.VideoWriter(name+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_shape,False)

for i in range(length):
        out.write(raw[i])

#Close
out.release()

cv2.destroyAllWindows()
