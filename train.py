import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
import wandb
import random
from tqdm import tqdm


# Define CNN model
class MyNet(nn.Module):
    
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList([
            nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1)
            for _ in range(args.nConv - 1)
        ])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(args.nChannel) for _ in range(args.nConv - 1)])
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# Load and preprocess an image
def load_image(image_path):
    im = cv2.imread(image_path)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    if use_cuda:
        data = data.cuda()
    return Variable(data), im


# Training function
def train_model(model, data, im):
	model.train()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

	# Define losses
	loss_fn = torch.nn.CrossEntropyLoss()
	loss_hpy = torch.nn.L1Loss()
	loss_hpz = torch.nn.L1Loss()

	HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel).cuda() if use_cuda else torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
	HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel).cuda() if use_cuda else torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)

	label_colours = np.array([
		(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0),
		(128, 0, 128), (0, 255, 255), (255, 192, 203), (128, 128, 0), (75, 0, 130),
		(255, 69, 0), (255, 20, 147), (0, 128, 128), (50, 205, 50), (138, 43, 226),
		(220, 20, 60), (240, 230, 140), (255, 105, 180), (154, 205, 50), (70, 130, 180),
		(244, 164, 96), (32, 178, 170), (255, 99, 71), (210, 105, 30), (127, 255, 212),
		(105, 105, 105), (192, 192, 192), (47, 79, 79), (255, 215, 0), (123, 104, 238),
		(173, 216, 230), (0, 191, 255), (184, 134, 11), (0, 255, 127), (255, 218, 185),
		(255, 160, 122), (46, 139, 87), (250, 128, 114), (255, 222, 173), (0, 206, 209),
		(233, 150, 122), (165, 42, 42), (255, 250, 205), (139, 69, 19), (153, 50, 204),
		(72, 209, 204), (147, 112, 219), (238, 130, 238), (255, 182, 193), (255, 0, 255),
		(0, 128, 0), (255, 248, 220), (250, 235, 215), (100, 149, 237), (255, 228, 196),
		(160, 82, 45), (127, 255, 0), (189, 183, 107), (0, 139, 139), (65, 105, 225),
		(139, 0, 0), (255, 140, 0), (85, 107, 47), (218, 165, 32), (95, 158, 160),
		(30, 144, 255), (72, 61, 139), (199, 21, 133), (178, 34, 34), (70, 130, 180),
		(255, 228, 225), (222, 184, 135), (128, 0, 0), (0, 255, 255), (173, 255, 47),
		(144, 238, 144), (0, 250, 154), (186, 85, 211), (250, 250, 210), (0, 0, 139),
		(255, 228, 181), (240, 128, 128), (0, 128, 128), (255, 255, 255), (240, 255, 240),
		(224, 255, 255), (245, 245, 220), (0, 0, 128), (240, 255, 255), (245, 255, 250),
		(47, 79, 79), (139, 0, 139), (176, 224, 230), (245, 245, 245), (255, 250, 240),
		(255, 255, 240), (255, 235, 205), (255, 239, 213), (253, 245, 230), (245, 222, 179),
	])
      
	for batch_idx in tqdm(range(args.maxIter), desc="Masking progress"):
		optimizer.zero_grad()
		output = model(data)[0]
		output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

		outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
		HPy = outputHP[1:, :, :] - outputHP[:-1, :, :]
		HPz = outputHP[:, 1:, :] - outputHP[:, :-1, :]
		
		lhpy = loss_hpy(HPy, HPy_target)
		lhpz = loss_hpz(HPz, HPz_target)
		
		ignore, target = torch.max(output, 1)
		im_target = target.data.cpu().numpy()
		nLabels = len(np.unique(im_target))

		# Visualization
		im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
		im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
		if args.visualize:
			cv2.imshow("output", im_target_rgb)
			cv2.waitKey(10)

		# Compute loss
		loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
		loss.backward()
		optimizer.step()

		# Log training metrics
		# wandb.log({"iteration": batch_idx, "loss": loss.item(), "nLabels": nLabels})
		# print(f"{batch_idx}/{args.maxIter} | labels: {nLabels} | loss: {loss.item()}")

		# Stop early if the number of labels reaches minimum
		if nLabels <= args.minLabels:
			print(f"Reached minLabels: {args.minLabels}. Stopping training.")
			break

	return im_target_rgb


# Process multiple images in a directory
def process_directory(input_dir):
	image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:20]
	model = MyNet(3)
	if use_cuda:
		model.cuda()
	for image_file in tqdm(image_files, desc="Images segmented"):
		print(f"Processing {image_file}")
		data, im = load_image(os.path.join(input_dir, image_file))
		segmented_image = train_model(model, data, im)
		output_path = os.path.join(input_dir, f"segmented_{image_file}")
		cv2.imwrite(output_path, segmented_image)
		print(f"Saved segmented image to {output_path}")
		
	# Save checkpoint
	torch.save(model.state_dict(), args.save_checkpoint)
	print(f"Checkpoint saved at {args.save_checkpoint}")



# Start processing
if __name__ == "__main__":
    
	use_cuda = torch.cuda.is_available()

	# Argument parser
	parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with Logging')
	parser.add_argument('--scribble', action='store_true', default=False, help='use scribbles')
	parser.add_argument('--nChannel', type=int, default=100, help='number of channels')
	parser.add_argument('--maxIter', type=int, default=1000, help='number of maximum iterations')
	parser.add_argument('--minLabels', type=int, default=50, help='minimum number of labels')
	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument('--nConv', type=int, default=2, help='number of convolutional layers')
	parser.add_argument('--visualize', type=int, default=1, help='visualization flag')
	parser.add_argument('--input_dir', type=str, required=True, help='input image directory')
	parser.add_argument('--stepsize_sim', type=float, default=1, help='step size for similarity loss')
	parser.add_argument('--stepsize_con', type=float, default=1, help='step size for continuity loss')
	parser.add_argument('--stepsize_scr', type=float, default=0.5, help='step size for scribble loss')
	parser.add_argument('--save_checkpoint', type=str, default='checkpoint.pth', help='path to save model checkpoints')
	parser.add_argument('--wandb_project', type=str, default='kanezaki-segmentation', help='WandB project name')
	args = parser.parse_args()

	# wandb.init(project=args.wandb_project, config=vars(args))

	process_directory(args.input_dir)
