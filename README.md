not much here yet, but just in case:

command for google colab to test model via python file:
==

# 1

```
import torch
print(f"CUDA available:{torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
```

# 2
```
!pip install -q unsloth trl peft accelerate
```

# 3
```
!python model_testing_v1.py
```
