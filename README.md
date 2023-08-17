# CEPROT
Here is the relevant code and data for our proposed novel approach CEPROT(Co-Evolution of Production-Test Code).
Data in master branch.
# CEPROT

Our model is divided into two stages, identification and generation. And we provide our model and three other baseline models here.

## Environment

In addition to SITAR, our operating environment is based on the Pytorch Framework.

CUDA 11.3 can be applied to most current graphics card driver versions. And you can use `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`to install pytorch-gpu. For other versions, you can see  [PyTorch](https://pytorch.org/) .

## Identify stage


### SITAR
SITAR consists of two processes, extracting the features of the method under the Parse folder and classifying them through four other classification models.

1. First your environment must contain JDK11 and MVN, SITAR is run via the MVN command.

2. And Our main interface is in the `SITAR/Parse/src/main/java/SITAR/MethodImplementation.java`, you only need to change the file path in it.

3. To run SITAR, you must know the Data Format in Method folder, and below is an example.

   SITAR/Method/project_1...project_n, and each of these project folders contains a certain number of json files, the json data format is as follows:

   ```
   {
     "classFile": "SimpleGroupMembershipResolverTest",
     "className": "SimpleGroupMembershipResolverTest",
     "packages": "org.apache.james.mailbox.acl",
     "methodInfos": [
   ​    {
   ​      "md5": "763bmIV1vpB5m7JLi/XOoA==",
   ​      "methodName": "setUp",
   ​      "parameters": "[]"
   ​    },
   ​    {
   ​      "md5": "ahJyEZ1F6A6Q8RAqr+O9Fw==",
   ​      "methodName": "isMemberShouldReturnFalseWhenEmptyResolver",
   ​      "parameters": "[]"
   ​    },
   ​    {
   ​      "md5": "E0/mNBmUnjUnlaoxrSTUcg==",
   ​      "methodName": "isMemberShouldReturnTrueWhenTheSearchedMembershipIsPresent",
   ​      "parameters": "[]"
   ​    },
   ​    {
   ​      "md5": "OrXailKIXnAmnc2aE7NXJQ==",
   ​      "methodName": "addMembershipShouldAddAMembershipWhenNonNullUser",
   ​      "parameters": "[]"
   ​    },
   ​    {
   ​      "md5": "mZTcTU/sVp0Uig3RfaqKBw==",
   ​      "methodName": "addMembershipShouldAddAMembershipWithANullUser",
   ​      "parameters": "[]"
   ​    }
     ],
     "addLines": [
   ​    1,
   ​    2,
   ​    3,
   .......,
   	78
     ],
     "delLines": [],
     "type": "ADD",
     "path": "mailbox/api/src/test/java/org/apache/james/mailbox/acl/SimpleGroupMembershipResolverTest.java",
     "pro_Commit": "476ddbf1e92c158ef2a33ba4f97a378790bbca4e",
     "pro_Path": "mailbox/api/src/main/java/org/apache/james/mailbox/acl/SimpleGroupMembershipResolver.java",
     "file_Date": "2015-10-07 21:14:19",
     "pro_Date": "2015-10-07 21:14:19",
     "commitID": "",
     "projectName": "james-project",
     "sha1": "f77dbbb31453474b9b1d515c6075fe5fc4635e61"
   }
   ```

   And the files located in "./Method/" should consist the following fields:

   - reposity: the project's name
   - Path: The test file path
   - Pro_Path: production file path
   - File_Date: test code commit date
   - Pro_Date: production code commit date
   - Pro_Commit: production code commit ID
   - Commit: Test code commit ID

   Production code and test code should follow **name convention**:
   for example: src/main/java -> src/test/java

   *Utility.getInstance.ReadInformation* store the project git location.

4. When you enter the project and download the dependencies of the Maven central repository, you can use   

```
mvn compile
mvn exec:java -Dexec.mainClass="src.main.java.SITAR.MethodImplementation"
```

 to run SITAR to extract the features. 

### KNN

1. The `gen_candidate.py` will generate  candidates for each case in the test set, and the results will be saved in `test_top.jsonl`.
2. The `classfiar.py` will classify the generated candidates, generate a label for each example in the test set and save it to a file.
3. Last, the `eval.py` will calculate each classification evaluation metrics, and save the TP and FP samples in the file.

### NMT

Due to the different data formats, we provide a data conversion script, you can convert the data in (`CEPROT/Updating/NMT/convert_data.py`). Also, since the NMT generation is concatenated with spaces, we also provide a script to convert the result, which is here(`code/updating/nmt/convert_readable.py`).

1. After you convert the data, you can adjust the parameters in `configs/OCD.yml`.
2. You can build your own vocabulary using `vocab.py`.
3. Then, run with `python -m main.py run_ocd  .configs/OCD.yml OCD `. And you can find all logs in the OCD folder.

### Our CEPROT

1. To run our indentification model, first you have to configure the file (configs.py). And  run with this command `python run_clone.py --do_train --do_eval --do_test`. During the evaluation process, TP_id and FP_id will be generated, please record them, which will be used in the second stage. 
2. And pay attention to each path in the `configs.py`.

## Update stage

### CodeBleu

The folder uesd for calculate accuracy and codebleu score for updating models. You can use `calc_code_bleu.py` and `cal_acc.py`.

Note: To use the script, you must install Tree-sitter([Tree-sitter｜Introduction](https://tree-sitter.github.io/tree-sitter/)).

### KNN

1. The `load_train_data.py` will generate  candidates for each case in the test set, and the results will be saved in `test_top.jsonl`.
2. The `cos_sim.py` will classify the generated candidates, generate a case for each example in the test set and save it to a file.
3. Last, you can infer PFC samples use `class_infer.py`. 

### NMT

As the same as first stage, due to the different data formats, we provide a data conversion script, you can convert the data in (`convert_data.py`). And you can build your environment use `environment.yml`.

1. You can build your own vocabulary using `vocab.py`.

2. You can use the following command to train.

   `python train.py --train-batch-size 8 --valid-batch-size 8 --max-epoch 25 --train-data  train.json --dev-data valid.json  --mix-vocab --vocab mix_vocab.json  --cuda  --input-feed --share-embed  --dropout 0.2 --use-pre-embed  --freeze-pre-embed --vocab-embed mix_vocab_embeddings.pkl --model-class models.updater.CoAttnBPBAUpdater  --log-dir CoAttnBPBAUpdater --save-to CoAttnBPBAUpdater/model.bin `

3. Then, use ` python infer.py ./CoAttnBPBAUpdater/model.bin  valid.json  --cuda ` for infer test cases. And the result will be saved `./CoAttnBPBAUpdater/result.jsonl`.
4. For evaling, use `python eval.py valid.json ./CoAttnBPBAUpdater/result.jsonl`
5. In the inference phase of FP samples, you can replace the valid set with FP samples, use `infer.py` for inference and then use `class_infer.py` to calculate FPC.

### Our CEPROT

1. During the update phase, you can run with `python train_codet5_edit_focal.py`. The premise is that you must configure your own path in the (codet5_edit_focal.py).
2. `python train_codet5_edit_focal.py `
3. The calculation of the two-stage metric is the same as NMT, you can just set do_test to true in the `codet5_edit_focal.py`.It's easy to find and run our programs.

