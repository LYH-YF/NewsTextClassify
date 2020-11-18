1. ### data download:
    run `sh data_download.sh`

2. ### data_process:
    run code in 'runner.ipynb'

3. ### train model:
    * 3.1 train `transformer` model:
        run `python trainer.py --model transformer --epoch 40 --batchsize 128`
    * 3.2 train `textcnn` model:
        run `python trainer.py --model textcnn --epoch 40 --batchsize 128 --embsize 5 `

4. ### test model:
    4.1 test transformer model:
        run `python tester.py --model transformer --batchsize 128`