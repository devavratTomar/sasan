from trainer import TrainerCycleGANMSMultiDSeg
import utilities.util as util
from collections import OrderedDict
from utilities import IterationCounter, Visualizer

from dataset import WholeHeartDatasetPostProcessed

from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from collections import namedtuple

os.environ["CUDA_VISIBLE_DEVICES"]="0"

colors = torch.tensor([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36]]) #AA

def color_seg(seg):
    out = []
    for s in seg:
        out.append((colors[s[0]]).permute(2, 0, 1))
    return torch.stack(out, dim=0)

def create_paths(opt):
    # create checkpoints directory
    util.ensure_dir(opt.checkpoints_dir)
    util.ensure_dir(os.path.join(opt.checkpoints_dir, 'models'))
    util.ensure_dir(os.path.join(opt.checkpoints_dir, 'tf_logs'))

def train_ms(opt):
    # B is mri, A is ct
    # dataset train unpaired
    dataset_B = WholeHeartDatasetPostProcessed(opt.dataroot_B, 'ct')   
    dataset_A = WholeHeartDatasetPostProcessed(opt.dataroot_A, 'mr')
    
    # test dataset unpaired
    dataset_B_test = WholeHeartDatasetPostProcessed(opt.dataroot_B_test, 'ct')   
    dataset_A_test = WholeHeartDatasetPostProcessed(opt.dataroot_A_test, 'mr')

    batch_size = opt.batch_size
        
    dataloader_A = DataLoader(dataset_A, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=opt.num_workers)
    dataloader_B = DataLoader(dataset_B, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=opt.num_workers)
    
    dataloader_A_test = DataLoader(dataset_A_test, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=opt.num_workers)
    dataloader_B_test = DataLoader(dataset_B_test, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=opt.num_workers)

    len_dataset = len(dataset_B)
    
    # create trainer to train the models
    trainer = TrainerCycleGANMSMultiDSeg(opt)
    
    # visualizer for tensorflow summary
    visualizer = Visualizer(opt)
        
    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len_dataset)
    
    
    # update batch_size
    iter_counter.set_batchsize(batch_size)
    
    print("Length of dataset: ", len_dataset)

    use_gpu = len(opt.gpu_ids) != 0

    for epoch in iter_counter.training_epochs():
        
        # update learning rates if number of total_epochs
        if epoch> iter_counter.total_epochs//2:
            lr = 2.0*opt.lr*(iter_counter.total_epochs + 1.0 - epoch)/(iter_counter.total_epochs + 2.0)
            trainer.update_learning_rate(lr)

        iter_counter.record_epoch_start(epoch)   
        
        A_iter = iter(dataloader_A)
        
        #start training for the current epoch, we finish epoch according
        for i, [data_i_B, seg_B] in enumerate(dataloader_B, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            data_i_A, seg_A = next(A_iter)

            if use_gpu:
                data_i_A, seg_A = data_i_A.cuda(), seg_A.cuda()
                data_i_B, seg_B = data_i_B.cuda(), seg_B.cuda()
            
            trainer.run_g_step(data_i_A, seg_A, data_i_B)
            trainer.run_d_step(data_i_A, seg_A, data_i_B)
            
            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                B_test_gen = []
                A_test_gen = []
                B_test_seg_gen = []
                A_test_seg_gen = []
                FakeA_test_seg_gen = []


                B_test = []
                A_test = []
                
                B_seg_test = []
                A_seg_test = []

                A_test_iter = iter(dataloader_A_test)
                
                for j, [img_test_B, seg_test_B] in enumerate(dataloader_B_test):
                    if j > 5:
                        break
                    img_test_A, seg_test_A = next(A_test_iter)

                    if use_gpu:
                        img_test_A = img_test_A.cuda()
                        img_test_B = img_test_B.cuda()
                    
                    fake_test_A, fake_test_B, pred_seg_A, pred_seg_B, pred_seg_fake_A = trainer.run_test(img_test_A, img_test_B)
                    B_test_gen.append(fake_test_B)
                    A_test_gen.append(fake_test_A)

                    B_test_seg_gen.append(pred_seg_B)
                    A_test_seg_gen.append(pred_seg_A)
                    FakeA_test_seg_gen.append(pred_seg_fake_A)
                    
                    B_test.append(img_test_B.cpu())
                    A_test.append(img_test_A.cpu())

                    B_seg_test.append(seg_test_B)
                    A_seg_test.append(seg_test_A)
                
                B_test_gen = torch.cat(B_test_gen, dim=0)
                A_test_gen = torch.cat(A_test_gen, dim=0)
                
                B_test_seg_gen = torch.cat(B_test_seg_gen, dim=0)
                A_test_seg_gen = torch.cat(A_test_seg_gen, dim=0)
                FakeA_test_seg_gen = torch.cat(FakeA_test_seg_gen, dim=0)
                
                A_test = torch.cat(A_test, dim=0)
                B_test = torch.cat(B_test, dim=0)
                A_seg_test = torch.cat(A_seg_test, dim=0)
                B_seg_test = torch.cat(B_seg_test, dim=0)

                #[:, 1:2, :, :]
                visuals = OrderedDict([
                    ('synthesized_A', trainer.fake_A[:, 1:2, :, :]),
                    ('real_A', data_i_A[:, 1:2, :, :]),
                    ('atten_A', trainer.attentions_A),
                    ('atten_B', trainer.attentions_B),
                    ('seg_A_pred', color_seg(trainer.seg_A.unsqueeze(1))),
                    ('seg_B_pred', color_seg(trainer.seg_B.unsqueeze(1))),
                    ('seg_fake_A_pred', color_seg(trainer.seg_fake_A.unsqueeze(1))),
                    ('seg_A', color_seg(seg_A.cpu().unsqueeze(1))),
                    ('seg_B', color_seg(seg_B.cpu().unsqueeze(1))),
                    ('synthesized_B', trainer.fake_B[:, 1:2, :, :]),
                    ('real_B', data_i_B[:, 1:2, :, :]),
                    ('test_real_A', A_test[:, 1:2, :, :]),
                    ('test_real_A_seg', color_seg(A_seg_test.unsqueeze(1))),
                    ('test_real_A_seg_pred', color_seg(A_test_seg_gen.unsqueeze(1))),
                    ('test_fake_A_seg_pred', color_seg(FakeA_test_seg_gen.unsqueeze(1))),
                    ('test_fake_A', A_test_gen[:, 1:2, :, :]),
                    ('test_real_B', B_test[:, 1:2, :, :]),
                    ('test_real_B_seg', color_seg(B_seg_test.unsqueeze(1))),
                    ('test_real_B_seg_pred', color_seg(B_test_seg_gen.unsqueeze(1))),
                    ('test_fake_B', B_test_gen[:, 1:2, :, :])])
                
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        iter_counter.record_epoch_end()
        
        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

            
    print('Training was finished successfully')
    

def main(dataroot_A=None, dataroot_A_test=None, dataroot_B=None, dataroot_B_test=None, checkpoints_dir=None):
    options = {
        'checkpoints_dir':checkpoints_dir,
        'checkpoints_pretrained_dir':'./checkpoints_pretrained',
        'dataroot_A':dataroot_A,
        'dataroot_B':dataroot_B,
        'dataroot_A_test':dataroot_A_test,
        'dataroot_B_test':dataroot_B_test,
        'nepochs':50,
        'batch_size':2, # testing only
        'num_workers':8,
        'gpu_ids':[0],
        'lr':1e-4,
        'output_ch':3,
        'input_ch':3,
        'save_epoch_freq':5,
        'save_latest_freq':500,
        'print_freq':10,
        'display_freq':100,
        'continue_train':False,
        'beta1':0.5,
        'lambda_c':10.0,
        'lambda_seg':0.1,
        'lambda_seg_p':0.1,
        'lambda_adver':2,
        'lambda_id':2.5,
        'lambda_orth':1.0,
        'tf_log':True,
        'gan_type':'lsgan',
        'n_attention':8,
        'nf':64,
        'n_classes':5,
        'mult':1.0
    }

    opt = namedtuple("Option", options.keys())(*options.values())
    create_paths(opt)
    #train(opt)
    train_ms(opt)

if __name__ == '__main__':
    main(
        '../whole_heart/mr_train_filtered/',
        '../whole_heart/mr_test_filtered/',
        '../whole_heart/ct_train_filtered/',
        '../whole_heart/ct_test_filtered/',
        './checkpoints_domain_adapt_new')
        #'./checkpoints_testing')
