def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios    = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
                             ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]
        self.scenarios_hp = [('10', '4'), ('8', '19'), ('4', '8')]
        self.private_classes = [{'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]}]
        self.generate_private = None  # initialized in Trainer
        self.da_method = None         # initialized in Trainer
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.src_balanced = True

        # ── Shared ────────────────────────────────────────────────────────────
        self.input_channels = 1
        self.final_out_channels = 128
        self.dropout = 0.2

        # ── CNN / S3Layer / RESNET18 ──────────────────────────────────────────
        self.kernel_size = 25
        self.stride = 6
        self.mid_channels = 64
        self.features_len = 1          # AdaptiveAvgPool1d target length

        # ── FNO ──────────────────────────────────────────────────────────────
        self.isFNO = False
        self.fourier_modes = 64        # number of low-frequency Fourier modes kept

        # ── TCN ──────────────────────────────────────────────────────────────
        self.tcn_layers = [32, 64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15
        self.tcn_dropout = 0.0

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # ── TimesNet ─────────────────────────────────────────────────────────
        self.isTimesNet = False
        self.e_layers = 1              # number of TimesBlocks
        self.d_model = 16             # model dimension after embedding
        self.embed = "fixed"
        self.freq = "h"
        self.enc_in = self.input_channels
        self.pred_len = 0
        self.d_ff = 64                 # Inception block inner dimension
        self.num_kernels = 6           # Inception block kernel count
        self.top_k = 3                 # top-k FFT periods

        # ── TSLANet ──────────────────────────────────────────────────────────
        self.patch_size = 8
        self.emb_dim = 128
        self.depth = 2                 # number of TSLANet layers
        self.masking_ratio = 0.4
        self.ICB = True                # use Interaction Conv Block
        self.ASB = True                # use Adaptive Spectral Block
        self.adaptive_filter = True    # adaptive high-freq mask inside ASB

        # ── Mamba ────────────────────────────────────────────────────────────
        self.mamba_d_model = 128       # hidden/model dimension
        self.mamba_d_state = 16        # SSM state dimension
        self.mamba_d_conv = 4          # local depthwise conv kernel size
        self.mamba_expand = 2          # inner dim expansion factor
        self.mamba_depth = 4           # number of MambaBlocks

        # ── PatchTST ─────────────────────────────────────────────────────────
        self.patchtst_patch_len = 16   # patch length
        self.patchtst_stride = 8       # patch stride (controls overlap)
        self.patchtst_embed_dim = 128  # embedding dimension
        self.patchtst_depth = 4        # number of Transformer layers
        self.patchtst_num_heads = 8    # attention heads
        self.patchtst_mlp_ratio = 4.0  # FFN hidden dim = embed_dim * mlp_ratio

        # ── Foundation models (Moment / Mantis / Chronos) ────────────────────
        self.fm_mlp_hidden_dims = [512, 256]  # hidden sizes of the trainable MLP head

        # ── Discriminator ─────────────────────────────────────────────────────
        self.disc_hid_dim = 100
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()
        # data parameters
        self.num_classes = 6
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128
        self.scenarios    = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"),
                             ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")]
        self.scenarios_hp = [('3', '4'), ('6', '7'), ('7', '8')]
        self.private_classes = [{'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]}]
        self.generate_private = None  # initialized in Trainer
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.src_balanced = True
        self.temp = 0.05

        # ── Shared ────────────────────────────────────────────────────────────
        self.input_channels = 3
        self.final_out_channels = 128
        self.dropout = 0.5

        # ── CNN / S3Layer / RESNET18 ──────────────────────────────────────────
        self.kernel_size = 5
        self.stride = 1
        self.mid_channels = 64
        self.features_len = 1          # AdaptiveAvgPool1d target length

        # ── FNO ──────────────────────────────────────────────────────────────
        self.isFNO = False
        self.fourier_modes = 64        # number of low-frequency Fourier modes kept

        # ── TCN ──────────────────────────────────────────────────────────────
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # ── TimesNet ─────────────────────────────────────────────────────────
        self.isTimesNet = False
        self.e_layers = 3              # number of TimesBlocks
        self.d_model = 32             # model dimension after embedding
        self.embed = "fixed"
        self.freq = "h"
        self.enc_in = self.input_channels
        self.pred_len = 0
        self.d_ff = 128                # Inception block inner dimension
        self.num_kernels = 6           # Inception block kernel count
        self.top_k = 5                 # top-k FFT periods

        # ── TSLANet ──────────────────────────────────────────────────────────
        self.patch_size = 8
        self.emb_dim = 128
        self.depth = 2                 # number of TSLANet layers
        self.masking_ratio = 0.4
        self.ICB = True                # use Interaction Conv Block
        self.ASB = True                # use Adaptive Spectral Block
        self.adaptive_filter = True    # adaptive high-freq mask inside ASB

        # ── Mamba ────────────────────────────────────────────────────────────
        self.mamba_d_model = 128       # hidden/model dimension
        self.mamba_d_state = 16        # SSM state dimension
        self.mamba_d_conv = 2          # local depthwise conv kernel size
        self.mamba_expand = 4          # inner dim expansion factor
        self.mamba_depth = 4           # number of MambaBlocks

        # ── PatchTST ─────────────────────────────────────────────────────────
        self.patchtst_patch_len = 16   # patch length
        self.patchtst_stride = 8       # patch stride (controls overlap)
        self.patchtst_embed_dim = 128  # embedding dimension
        self.patchtst_depth = 4        # number of Transformer layers
        self.patchtst_num_heads = 8    # attention heads
        self.patchtst_mlp_ratio = 4.0  # FFN hidden dim = embed_dim * mlp_ratio

        # ── Foundation models (Moment / Mantis / Chronos) ────────────────────
        self.fm_mlp_hidden_dims = [512, 256]  # hidden sizes of the trainable MLP head

        # ── Discriminator ─────────────────────────────────────────────────────
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class HAR():
    def __init__(self):
        super(HAR, self)
        # data parameters
        self.num_classes = 5
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.scenarios    = [("6", "23"), ("9", "18"), ("12", "16"), ("24", "8"), ("30", "20"),
                             ("13", "3"), ("15", "21"), ("1", "14"), ("17", "29"), ("22", "4")]
        self.scenarios_hp = [("2", "11"), ("18", "27"), ("20", "5"), ("7", "13"), ("28", "27")]
        #self.scenarios_hp = [("2", "11"), ("18", "27"), ("20", "5"), ("7", "13"), ("28", "27")]
        self.private_classes = [{'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]},
                                {'src': [2], 'trg': [3]}, {'src': [2], 'trg': [3]}]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.src_balanced = True
        self.temp = 0.05

        # ── Shared ────────────────────────────────────────────────────────────
        self.input_channels = 9
        self.final_out_channels = 128
        self.dropout = 0.5

        # ── CNN / S3Layer / RESNET18 ──────────────────────────────────────────
        self.kernel_size = 5
        self.stride = 1
        self.mid_channels = 64
        self.features_len = 1          # AdaptiveAvgPool1d target length

        # ── FNO ──────────────────────────────────────────────────────────────
        self.isFNO = False
        self.fourier_modes = 64        # number of low-frequency Fourier modes kept

        # ── TCN ──────────────────────────────────────────────────────────────
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # ── TimesNet ─────────────────────────────────────────────────────────
        self.isTimesNet = False
        self.e_layers = 3             # number of TimesBlocks
        self.d_model = 32             # model dimension after embedding
        self.embed = "fixed"
        self.freq = "h"
        self.enc_in = self.input_channels
        self.pred_len = 0
        self.d_ff = 128                # Inception block inner dimension
        self.num_kernels = 6           # Inception block kernel count
        self.top_k = 5                 # top-k FFT periods

        # ── TSLANet ──────────────────────────────────────────────────────────
        self.patch_size = 8
        self.emb_dim = 128
        self.depth = 2                 # number of TSLANet layers
        self.masking_ratio = 0.4
        self.ICB = True                # use Interaction Conv Block
        self.ASB = True                # use Adaptive Spectral Block
        self.adaptive_filter = True    # adaptive high-freq mask inside ASB

        # ── Mamba ────────────────────────────────────────────────────────────
        self.mamba_d_model = 128       # hidden/model dimension
        self.mamba_d_state = 16        # SSM state dimension
        self.mamba_d_conv = 4          # local depthwise conv kernel size
        self.mamba_expand = 2          # inner dim expansion factor
        self.mamba_depth = 4           # number of MambaBlocks

        # ── PatchTST ─────────────────────────────────────────────────────────
        self.patchtst_patch_len = 16   # patch length
        self.patchtst_stride = 8       # patch stride (controls overlap)
        self.patchtst_embed_dim = 128  # embedding dimension
        self.patchtst_depth = 4        # number of Transformer layers
        self.patchtst_num_heads = 8    # attention heads
        self.patchtst_mlp_ratio = 4.0  # FFN hidden dim = embed_dim * mlp_ratio

        # ── Foundation models (Moment / Mantis / Chronos) ────────────────────
        self.fm_mlp_hidden_dims = [512, 256]  # hidden sizes of the trainable MLP head

        # ── Discriminator ─────────────────────────────────────────────────────
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128
