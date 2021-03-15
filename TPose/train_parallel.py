import torch
import sys
import numpy as np
import itertools
from T_Pose_model import *
from load_newfeature import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.parallel
import argparse
import time
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/train_features", help="Path to input dataset")
    parser.add_argument("--save_path", type=str, default="./model_checkpoints", help="Path to save")
    parser.add_argument("--num_epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=60, help="Size of each training batch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")

    opt = parser.parse_args()
    print(opt)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define training set
    train_dataset = TorchDataset(istrain=True, fea_dir=opt.dataset_path, isaug = True, repeat=1)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=32,pin_memory=True)

    # Define test set
    test_dataset = TorchDataset(istrain=False, fea_dir=opt.dataset_path, repeat=1)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=32,pin_memory=True)

    # Classification criterion
    #cls_criterion = nn.CrossEntropyLoss().to(device)
    cls_criterion = LabelSmoothingCrossEntropy().cuda()
    # Define network
    model =T_Pose_model(frames_number=60,joints_number=33,
        n_classes=226
    )

    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model,map_location='cuda:0'))
    else:
        model.init_weights()

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    #model = model.module
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

    def test_model(epoch,global_acc1,needsave1):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(test_dataloader):
            fea_sequences = Variable(X.to(device), requires_grad=False)
            y = y.type(torch.long)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # Get sequence predictions
                predictions = model(fea_sequences)
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            

            loss = cls_criterion(predictions, labels).item()
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Log test performance
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
        newacc =  np.mean(test_metrics["acc"])
        model.train()
        print("")
        return newacc
    global_acc = 0
    needsave = False
    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch} ---")
        for batch_i, (X, y) in enumerate(train_dataloader):

            if X.size(0) == 1:
                continue
            if epoch==50:
                optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr*0.1, weight_decay=0.0)
            if epoch==250:
                optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr*0.01, weight_decay=0.0)

            fea_sequences = Variable(X.to(device), requires_grad=True)
   
            y = y.type(torch.long)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Get sequence predictions
            predictions = model(fea_sequences)

            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()
            #optimizer.module.step()
            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    time_left,
                )
            )

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate the model on the test set
        newacc = test_model(epoch,global_acc,needsave)
        
        
        # Save model checkpoint
        if global_acc<newacc:#epoch % opt.checkpoint_interval == 0:
            os.makedirs(opt.save_path, exist_ok=True)
            torch.save(model.module.state_dict(), f"{opt.save_path}/T_Pose_model_{epoch}_{newacc}.pth")
            sys.stdout.write(
                "save model at %f"
                % (newacc)
            )
            global_acc = newacc
