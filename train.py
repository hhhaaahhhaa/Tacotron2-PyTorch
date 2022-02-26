import os
import time
import torch
import argparse
import numpy as np
# from inference import infer
from utils.util import mode
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.logger import Tacotron2Logger
from utils.dataset import ljdataset, LibriTTSDataset, ljcollate, librittscollate
from model.model import Tacotron2Loss
from model.tacotron import Tacotron2
np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)


def prepare_LJSpeech_dataloaders(fdir, preprocessed_data_dir):
	trainset = ljdataset(fdir, f"{preprocessed_data_dir}/train.txt")
	collate_fn = ljcollate(hps.n_frames_per_step)
	train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = True,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return {"train": train_loader, "val": None, "test": None}


def prepare_LibriTTS_dataloaders(fdir, preprocessed_data_dir):
	trainset = LibriTTSDataset(fdir, f"{preprocessed_data_dir}/train.txt")
	valset = LibriTTSDataset(fdir, f"{preprocessed_data_dir}/dev.txt")
	testset = LibriTTSDataset(fdir, f"{preprocessed_data_dir}/test.txt")
	collate_fn = librittscollate(hps.n_frames_per_step)
	train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = True,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	val_loader = DataLoader(valset, num_workers = hps.n_workers, shuffle = False,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = False, collate_fn = collate_fn)
	test_loader = DataLoader(testset, num_workers = hps.n_workers, shuffle = False,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = False, collate_fn = collate_fn)
	return {"train": train_loader, "val": val_loader, "test": test_loader}


def load_checkpoint(ckpt_pth, model, optimizer):
	ckpt_dict = torch.load(ckpt_pth)
	model.load_state_dict(ckpt_dict['model'])
	optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def train(args):
	# build model
	model = Tacotron2(args.preprocessed_data_dir)
	mode(model, True)
	optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr,
								betas = hps.betas, eps = hps.eps,
								weight_decay = hps.weight_decay)
	criterion = Tacotron2Loss()
	
	# load checkpoint
	iteration = 1
	if args.ckpt_pth != '':
		model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer)
		iteration += 1 # next iteration is iteration+1
	
	# get scheduler
	if hps.sch:
		lr_lambda = lambda step: hps.sch_step**0.5*min((step+1)*hps.sch_step**-1.5, (step+1)**-0.5)
		if args.ckpt_pth != '':
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
		else:
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
	
	# make dataset
	# loaders = prepare_LJSpeech_dataloaders(args.data_dir, args.preprocessed_data_dir)
	loaders = prepare_LibriTTS_dataloaders(args.data_dir, args.preprocessed_data_dir)
	train_loader = loaders["train"]
	
	# get logger ready
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir):
			os.makedirs(args.log_dir)
			os.chmod(args.log_dir, 0o775)
		logger = Tacotron2Logger(args.log_dir)

	os.makedirs(args.result_dir, exist_ok=True)

	# get ckpt_dir ready
	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
		os.chmod(args.ckpt_dir, 0o775)
	
	model.train()
	# ================ MAIN TRAINNIG LOOP! ===================
	while iteration <= hps.max_iter:
		for batch in train_loader:
			if iteration > hps.max_iter:
				break
			start = time.perf_counter()
			x, y = model.parse_batch(batch)
			y_pred = model(x)

			# loss
			loss, item = criterion(y_pred, y, iteration)
			
			# zero grad
			model.zero_grad()
			
			# backward, grad_norm, and update
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)
			optimizer.step()
			if hps.sch:
				scheduler.step()
			
			# info
			dur = time.perf_counter()-start
			print('Iter: {} Loss: {:.2e} Grad Norm: {:.2e} {:.1f}s/it'.format(
				iteration, item, grad_norm, dur))
			
			# log
			if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
				learning_rate = optimizer.param_groups[0]['lr']
				logger.log_training(item, grad_norm, learning_rate, iteration)
			
			# sample
			if args.log_dir != '' and (iteration % hps.iters_per_sample == 0):
				model.eval()
				output = model.infer(hps.eg_text, "103")
				model.train()
				logger.sample_training(output, iteration)
				logger.save_audio(output, f"{args.result_dir}/{iteration:07d}.wav")
			
			# save ckpt
			if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
				ckpt_pth = os.path.join(args.ckpt_dir, 'ckpt_{}'.format(iteration))
				save_checkpoint(model, optimizer, iteration, ckpt_pth)

			iteration += 1
	if args.log_dir != '':
		logger.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# path
	parser.add_argument('-d', '--data_dir', type = str,
						help = 'directory to load data')
	parser.add_argument('-pd', '--preprocessed_data_dir', type = str,
						help = 'directory of preprocessed files')
	parser.add_argument('-rd', '--result_dir', type = str, default = 'result',
						help = 'directory to save results')
	parser.add_argument('-l', '--log_dir', type = str, default = 'log',
						help = 'directory to save tensorboard logs')
	parser.add_argument('-cd', '--ckpt_dir', type = str, default = 'ckpt',
						help = 'directory to save checkpoints')
	parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
						help = 'path to load checkpoints')

	args = parser.parse_args()
	
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False # faster due to dynamic input shape
	train(args)
