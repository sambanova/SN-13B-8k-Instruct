# SN-13B-8k-Instruct
This repo contains the reproducibility information for the numbers listed in the SN-13B-8k-Instruct blogpost.  Scrolls and ZeroScrolls refer to the following benchmarks:
1. [Scrolls Benchmark](https://www.scrolls-benchmark.com/)
2. [ZeroScrolls Benchmark](https://www.zero.scrolls-benchmark.com/) 

## Setup Eleuther AI LM Evaluation Harness
1. git clone https://github.com/EleutherAI/lm-evaluation-harness.git
2. Checkout the commit of LM Evaluation Harness that we used to collect the results:
```
git checkout fe803c2920a85f6afb74ea05d1d2f98ec27f1a63`
```
3. Follow the setup instructions specified in the repository's README.

## ZeroScrolls Reproducibility
1. Add [ZeroScrolls task code](zero_scrolls.py) to the LM Evaluation Harness.
   - This will involve importing the zero scrolls tasks in the `tasks/__init__.py` file in LM Evaluation Harness.  You will need to add the following line to the `TASK_REGISTRY`:
   ```python
   **zero_scrolls.construct_tasks(),
   ```
2. Install [requirements](requirements.txt)
```
pip install requirements.txt
```
3. Run the following command in the LM Evaluation Harness:
```
python main.py --batch_size 1 --tasks zero_scrolls_gov_report,zero_scrolls_summ_screen_fd,zero_scrolls_qm_sum,zero_scrolls_squality,zero_scrolls_qasper,zero_scrolls_narrative_qa,zero_scrolls_quality,zero_scrolls_musique,zero_scrolls_space_digest,zero_scrolls_book_sum_sort --model gpt2 --model_args pretrained=sambanovasystems/SN-13B-8k-Instruct,dtype=float16 --num_fewshot 0 --no_cache
```


## Scrolls Reproducibility
1. In the LM Evaluation Harness, open `tasks/scrolls.py` and replace the `'\n'` with your model's end of text token in the `until` list for all `greedy_until` requests.
2. Run the following command in the LM Evaluation Harness:
```
python main.py --batch_size 1 --tasks scrolls_govreport,scrolls_qmsum,scrolls_quality,scrolls_summscreenfd --model gpt2 --model_args pretrained=sambanovasystems/SN-13B-8k-Instruct,dtype=float16 --num_fewshot 0  --no_cache
```
