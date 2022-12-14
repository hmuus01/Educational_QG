# Educational_QG

**This project focuses on the use of automatic question generation (QG) to generate educational
questions for reading practice and evaluation that are valuable for learners**

| **Main Files** |**Description** |
| --- | --- |
|`fine-tune_model_leaf.py`| This file contains the steps taken to **fine-tune** the models using the SQuAD dataset |
|`sciq_only_fine-tune_model.py`| This file contains the steps taken to **fine-tune** the models using the SCiQ dataset |
|`race_only_fine_tune_model.py`| This file contains the steps taken to **fine-tune** the models using the RACE dataset |
|||
|`run_t5_mlm.py`|This file contains the code adapted and modified from huggingface to **pre-train** a model |
|`utils_qa.py`|This file is associated with the 'run_t5_mlm' file and is also adapted from huggingface|
|`fromcheckpointflax_t5_cs_big`|This file in the `stratch` folder contains the **flags** used to pre-train the models|
|||
|`evaluate.py`|This file contains the steps taken to **evaluate** the models using the three datasets|
|||
|`spark.py`|This file contains the steps taken to **pre-process** the science and computer science data used to pre-train the model|
|||
|`flaskapp.py` | This file serves as the client, it is made with flask and loads the questions generated by the model which is then displayed to the user. | ✔ |

Training took place on blaze (UCL CS Linux Machine)



**Please Note** the trained models could not be uploaded due to size restrictions. 