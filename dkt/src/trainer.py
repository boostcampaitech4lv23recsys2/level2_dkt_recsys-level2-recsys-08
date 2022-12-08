import math
import os

import torch
import wandb
import mlflow

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert
from .lqtransformer import LQTransformer
from .lgcn_lqtransfomer import lightGCN_LQTransformer
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def run(args, train_data, valid_data, model):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log(
            {
                "epoch": epoch,
                "train_loss_epoch": train_loss,
                "train_auc_epoch": train_auc,
                "train_acc_epoch": train_acc,
                "valid_auc_epoch": auc,
                "valid_acc_epoch": acc,
            },step = epoch
        )

        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                "model.pt",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break
        wandb.run.summary['best_auc'] = best_auc
        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)

        mlflow.log_metric("VAL AUC",best_auc)
        mlflow.log_metric("VAL ACC",acc)
        mlflow.log_metric("TRAIN AUC",train_auc)
    mlflow.pytorch.log_model(model, artifact_path="model") # 모델 기록


def train(train_loader, model, optimizer, scheduler, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch))) #[6,64,20] 의 6 : [test, question, tag, correct, mask, interaction]
        preds = model(input) #[64,20]
        ## 구버전 lqtransformer 쓸 때 아래 3줄 사용 -> lgcn_lqtransfomer.py를 위해 다시 추가
        if args.model == 'lgcnlqt':
            targets = input[3][:,-1].unsqueeze(1)
        else:
            targets = input[3]  # correct #[64,20]

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach()) #detach는 clone()같은 복사방법중 한방법. 기존 tensor에서 gradient 전파가 안되는 텐서 생성이라 한다.
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)
        targets = input[3]  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data, model):

    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)

        # predictions
        preds = preds[:, -1]
        preds = torch.nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "submission_lgcnlqt_실험12.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == "lqtransformer":
        model = LQTransformer(args)
    if args.model == "lgcnlqt":
        model = lightGCN_LQTransformer(args)
        
    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch):

    #🙂7. FE할 때 여기 고치세요! 주의할 점 : 6번과정과 비슷한데, 끝에 mask 추가해주세요!
    # test, question, tag, correct, big, mid, problem, month, dayname, month_mean, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum, user_correct_answer, user_total_answer, user_acc, mask = batch
    test, question, tag, correct, big_category, mid_category, problem_num, \
    assIdx, month, day, hour, dayname, time_category, solvecumsum_category, \
    solvesec_3600, test_mean, test_std, tag_mean, tag_std, big_mean, big_std, user_correct_answer, user_total_answer, user_acc, \
    solvesec_cumsum,  big_category_cumconut, big_category_user_acc, big_category_user_std, big_category_answer, big_category_answer_log1p, elo_assessmentItemID, mask = batch

    # change to float
    mask = mask.float()
    correct = correct.float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    #🙂8. FE할 때 여기 고치세요! 주의할 점 : answerCode를 나타내는 correct와 mask는 빼고 해주세요!
    # # 다른 columns도 masking하고, masking한 0과 실제 0의 값을 구분위해+1        
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()
    big_category = ((big_category + 1) * mask).int()
    mid_category= ((mid_category+ 1) * mask).int()
    problem_num = ((problem_num + 1) * mask).int() 
    assIdx = ((assIdx + 1) * mask).int()
    month = ((month + 1) * mask).int()
    day = ((day + 1) * mask).int()
    hour = ((hour + 1) * mask).int()
    dayname = ((dayname + 1) * mask).int()
    time_category = ((time_category + 1) * mask).int()
    solvecumsum_category = ((solvecumsum_category + 1) * mask).int()
    # month_mean = ((month_mean + 1) * mask).int()
    solvesec_3600 = ((solvesec_3600 + 1) * mask).int()
    test_mean = ((test_mean + 1) * mask).int()
    test_std = ((test_std + 1) * mask).int()
    # test_sum = ((test_sum + 1) * mask).int()
    tag_mean = ((tag_mean + 1) * mask).int()
    tag_std = ((tag_std + 1) * mask).int()
    # tag_sum = ((tag_sum + 1) * mask).int()
    big_mean = ((big_mean + 1) * mask).int()
    big_std = ((big_std + 1) * mask).int()
    # big_sum = ((big_sum + 1) * mask).int()
    user_correct_answer = ((user_correct_answer + 1) * mask).int()
    user_total_answer = ((user_total_answer + 1) * mask).int()
    user_acc = ((user_acc + 1) * mask).int()
    # user_tag_cluster = ((user_tag_cluster + 1) * mask).int()
    solvesec_cumsum = ((solvesec_cumsum + 1) * mask).int()
    big_category_cumconut = ((big_category_cumconut + 1) * mask).int()
    big_category_user_acc = ((big_category_user_acc + 1) * mask).int()
    big_category_user_std = ((big_category_user_std + 1) * mask).int()
    big_category_answer = ((big_category_answer + 1) * mask).int()
    big_category_answer_log1p = ((big_category_answer_log1p + 1) * mask).int()
    elo_assessmentItemID = ((elo_assessmentItemID + 1) * mask).int()
    
    # new_feature = ((new_feature + 1) * mask).int()
    
    #🙂9. FE할 때 여기 고치세요! 주의할 점 : 7번과정과 비슷한데, 끝에 interaction을 붙여주세요!
    #👍여기까지 하셨다면, model에 넣기 전 피처추가 과정은 완료되었습니다. 이제 사용하실 모델에서 추가한 피처에 대해 임베딩하고 쓰시면 될겁니다!
    # return (test, question, tag, correct, mask, interaction, big, mid, problem, month, dayname, month_mean, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum, user_correct_answer, user_total_answer, user_acc)
    return (test, question, tag, correct, mask, interaction, big_category, mid_category, problem_num, 
            assIdx, month, day, hour, dayname, time_category, solvecumsum_category,
            solvesec_3600, test_mean, test_std, tag_mean, tag_std, big_mean, big_std, user_correct_answer, user_total_answer, user_acc,
            solvesec_cumsum,  big_category_cumconut, big_category_user_acc, big_category_user_std, big_category_answer, big_category_answer_log1p, elo_assessmentItemID)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
