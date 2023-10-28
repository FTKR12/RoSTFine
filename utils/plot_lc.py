import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot(args, train_log, valid_log, name):

    train_mean = np.mean(train_log, axis=0)
    valid_mean = np.mean(valid_log, axis=0)
    train_std = np.std(train_log, axis=0)
    valid_std = np.std(valid_log, axis=0)
    
    plt.figure(figsize=(6.4, 4.8))
    plt.title(args.model_name)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.ylim([0, 0.16])
    ## average score plot
    plt.plot(range(args.epochs), train_mean, label="train loss", color="tab:blue")
    plt.plot(range(args.epochs), valid_mean, label="vali loss", color="tab:orange")
    ## range std 
    plt.fill_between(range(args.epochs), train_mean-train_std, train_mean+train_std, color="tab:blue", alpha=0.2)
    plt.fill_between(range(args.epochs), valid_mean-valid_std, valid_mean+valid_std, color="tab:orange", alpha=0.2)
    plt.legend()
    plt.savefig(f'{args.output_dir}/learning_curves/learning_curve_{name}_all.png')
    plt.close()

def plot_for_rostfine(args, train_log, valid_log, name):
    train_mean, train_std = {}, {}
    valid_mean, valid_std = {}, {}
    for k,v in train_log.items():
        train_mean[k] = np.mean(train_log[k], axis=0)
        valid_mean[k] = np.mean(valid_log[k], axis=0)
        train_std[k] = np.std(train_log[k], axis=0)
        valid_std[k] = np.std(valid_log[k], axis=0)
    
    plt.figure(figsize=(6.4, 4.8))
    plt.title(args.model_name)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    #plt.ylim([0, 0.16])
    for k in train_mean.keys():
        # average score plot
        plt.plot(range(args.epochs), train_mean[k], label=f"train_{k}")
        plt.plot(range(args.epochs), valid_mean[k], label=f"valid_{k}")
        # range std 
        plt.fill_between(range(args.epochs), train_mean[k]-train_std[k], train_mean[k]+train_std[k], alpha=0.2)
        plt.fill_between(range(args.epochs), valid_mean[k]-valid_std[k], valid_mean[k]+valid_std[k], alpha=0.2)
    plt.legend()
    plt.savefig(f'{args.output_dir}/learning_curves/learning_curve_{name}_all.png')
    plt.close()

def plot_all(args):
    train_loss_log, valid_loss_log = [], []
    train_div_log, valid_div_log = [], []
    for fold in range(args.kfold):              
        if args.model_name == "rostfine":
            with open(f'{args.output_dir}/loss/log_div_{fold}.pkl', 'rb') as f:
                div_cv = pickle.load(f)
                train_div_log.append(div_cv['train div'])            
                valid_div_log.append(div_cv['valid div'])
        with open(f'{args.output_dir}/loss/log_loss_{fold}.pkl', 'rb') as f:
            loss_cv = pickle.load(f)
            train_loss_log.append(loss_cv['train loss'])            
            valid_loss_log.append(loss_cv['valid loss'])

    if args.model_name == "rostfine":
        train_loss_cv = {}
        train_loss_cv["vg"] = [x["vg"] for x in train_loss_log]
        train_loss_cv["vs"] = [x["vs"] for x in train_loss_log]
        train_loss_cv["vt"] = [x["vt"] for x in train_loss_log]
        train_loss_cv["all"] = [x["all"] for x in train_loss_log]
        train_loss_cv["avg"] = [x["avg"] for x in train_loss_log]
        valid_loss_cv = {}
        valid_loss_cv["vg"] = [x["vg"] for x in valid_loss_log]
        valid_loss_cv["vs"] = [x["vs"] for x in valid_loss_log]
        valid_loss_cv["vt"] = [x["vt"] for x in valid_loss_log]
        valid_loss_cv["all"] = [x["all"] for x in valid_loss_log]
        valid_loss_cv["avg"] = [x["avg"] for x in valid_loss_log]
        plot_for_rostfine(args, train_loss_cv, valid_loss_cv, "loss")

        train_div_cv = {}
        train_div_cv["gs"] = [x["gs"] for x in train_div_log]
        train_div_cv["gt"] = [x["gt"] for x in train_div_log]
        train_div_cv["st"] = [x["st"] for x in train_div_log]
        train_div_cv["avg"] = [x["avg"] for x in train_div_log]
        valid_div_cv = {}
        valid_div_cv["gs"] = [x["gs"] for x in valid_div_log]
        valid_div_cv["gt"] = [x["gt"] for x in valid_div_log]
        valid_div_cv["st"] = [x["st"] for x in valid_div_log]
        valid_div_cv["avg"] = [x["avg"] for x in valid_div_log]
        plot_for_rostfine(args, train_div_cv, valid_div_cv, "div")
        
    else:
        plot(args, train_loss_log, valid_loss_log, 'loss')
            