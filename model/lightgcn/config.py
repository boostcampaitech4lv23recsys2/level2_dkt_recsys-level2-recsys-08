# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="lightGCN",entity = 'recsys8')

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"
    
    # item_node
    itemnode = "assessmentItemID"

    # build
    embedding_dim = 128 if itemnode == 'assessmentItemID' else 256  # int
    num_layers = 1 if itemnode == 'assessmentItemID' else 2 # int
    alpha = None  # Optional[Union[float, Tensor]] # None이면 디폴트로 alpha = 1. / (num_layers + 1) 이다.
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    n_epoch = 30 # 30
    learning_rate = 0.09335 if itemnode == 'assessmentItemID' else 0.08249 # 0.001 -> 0.1로 수정
    early_stopping = 10
    weight_basepath = "./weight"


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
