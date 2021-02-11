from trainer import SegmentationTrainer, TrainerAttention
import utilities.util as util
from collections import OrderedDict
from utilities import IterationCounter, Visualizer

from dataset import WholeHeartDatasetPostProcessed

from torch.utils.data import DataLoader
import torch
import os

from collections import namedtuple
import sys

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

def train(opt):
    # batch_size
    batch_size = opt.batch_size

    # training dataset
    dataset = WholeHeartDatasetPostProcessed(opt.dataroot, opt.modality)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=opt.num_workers)

    # test dataset
    dataset_test = WholeHeartDatasetPostProcessed(opt.dataroot_test, opt.modality)
    dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=10, drop_last=True, num_workers=opt.num_workers)

    len_dataset = len(dataset)

    # create trainer object
    if 'attention' in opt.mode:
        print('Pre-training attention model')
        trainer = TrainerAttention(opt)
    else:
        trainer = SegmentationTrainer(opt)
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

        for i, [x, seg] in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            # load dataset to gpu
            if use_gpu:
                x, seg = x.cuda(), seg.cuda()

            # run backward step
            trainer.run_train_step(x, seg)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.losses
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                predictions = []
                gt_labels   = []
                input_imgs  = []

                for j, [x_test, seg_test] in enumerate(dataloader_test):
                    if j > 1:
                        break
                    if use_gpu:
                        x_test = x_test.cuda()
                        predict = trainer.run_test(x_test)
                        x_test = x_test.cpu()
                    
                    predictions.append(predict)
                    gt_labels.append(seg_test)
                    input_imgs.append(x_test)
                
                predictions = torch.cat(predictions, dim=0)
                gt_labels   = torch.cat(gt_labels, dim=0).unsqueeze(1)
                input_imgs  = torch.cat(input_imgs, dim=0)

                visuals = OrderedDict([
                    ('train_input_imgs', x),
                    ('train_gt_seg', color_seg(seg.unsqueeze(1))),
                    ('train_pred_seg', color_seg(trainer.predictions_index)),
                    #('train_attentions', trainer.attentions),
                    ('test_input_imgs', input_imgs),
                    ('test_gt_seg', color_seg(gt_labels)),
                    ('test_pred_seg', color_seg(predictions))])
                
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
    
    trainer.save('latest')
    print('Training was finished successfully')

def main(dataroot=None, dataroot_test=None, checkpoints_dir=None, modality=None, mode=None):
    options = {
        'modality':modality,
        'mode':mode,
        'checkpoints_dir':checkpoints_dir,
        'dataroot':dataroot,
        'dataroot_test':dataroot_test,
        'nepochs':2,
        'batch_size':16, # testing only
        'num_workers':8,
        'gpu_ids':[0],
        'lr':4*1e-4,
        'output_ch':3,
        'input_ch':3,
        'save_epoch_freq':1,
        'save_latest_freq':10000,
        'print_freq':10,
        'display_freq':1000,
        'continue_train':False,
        'beta1':0.9,
        'lambda_ce':1.0,
        'tf_log':True,
        'n_classes':5,
        'n_attention':8,
        'segmenter':'UNET'
    }

    opt = namedtuple("Option", options.keys())(*options.values())
    create_paths(opt)
    train(opt)

if __name__ == '__main__':
    mode = sys.argv[1]
    print(mode)
    
    if mode == 'mr':
        main(
            ['../whole_heart/mr_train_filtered/', '../whole_heart/fake_mr/'],
            '../whole_heart/mr_test_filtered/',
            './checkpoints_segmentation_mr',
            'mr',
            mode)
    elif mode == 'ct':
        main(
            ['../whole_heart/ct_train_filtered/', '../whole_heart/fake_ct/'],
            '../whole_heart/ct_test_filtered/',
            './checkpoints_segmentation_ct',
            'ct',
            mode)
    elif mode == 'attention_mr':
        main(
            '../whole_heart/mr_train_filtered/',
            '../whole_heart/mr_test_filtered/',
            './checkpoints_attention_mr',
            'mr',
            mode)
    elif mode == 'attention_ct':
        main(
            '../whole_heart/ct_train_filtered/',
            '../whole_heart/ct_test_filtered/',
            './checkpoints_attention_ct',
            'mr',
            mode)     

    else:
        print("Unidentified mode")
