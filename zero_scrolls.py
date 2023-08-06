from huggingface_hub import hf_hub_download
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from pathlib import Path
from typing import Callable, List, Optional
from abc import abstractmethod
import datasets
import jsonlines
import numpy as np
import os, shutil
import re
import multiprocessing as mp


def _process_doc_prepended_question(doc):
    # "When a query is given in addition to the raw text (as
    # in QMSum, Qasper, NarrativeQA, QuALITY, and ContractNLI),
    # we prepend it to the text, using two newlines as a natural separator"
    input_ = doc["input"]
    split = input_.find("\n\n")
    return {
        "id": doc["id"],
        "pid": doc["pid"],
        "input": input_,
        "output": doc["output"],
        "question": input_[0:split],
        "text": input_[split + 2:]
    }

# copied from here: https://huggingface.co/datasets/tau/scrolls/blob/main/metrics/scrolls.py
def download_metric() -> str:
    """Download the scrolls metric from huggingface.  This function was copied from here:
    https://huggingface.co/datasets/tau/scrolls/blob/main/metrics/scrolls.py
    """
    scrolls_metric_path = hf_hub_download(repo_type='dataset', repo_id="tau/zero_scrolls", filename="metrics/zero_scrolls.py")
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path


def get_scrolls_metric(dataset_name: str) -> Callable:
    """Get the specific scrolls metric based on the dataset name."""
    metrics_path = download_metric()
    return datasets.load_metric(metrics_path, dataset_name)


class ZeroScrollsTask(Task):
    """Base class for all tasks from the ZeroSCROLLS benchmark, see paper https://arxiv.org/pdf/2305.14196.pdf.  The
    ZeroSCROLLS benchmark is a suite of tasks that require reasoning over long text data.  This benchmark is used to
    evaluate our Long sequence size (>= 8k SS) models.
    """
    DATASET_PATH = 'tau/zero_scrolls'
    VERSION = 1

    def __init__(self,
                 task_name: str,
                 metric_names: List[str],
                 max_gen_toks: Optional[int] = None,
                 num_test_examples: Optional[str] = None,
                 predictions_output_path: Optional[str] = None,
                 *args, **kwargs):
        """Create a ZeroScrollsTask

        Args:
            task_name:  The name of the task.
            num_test_examples:  The number of test examples to evaluate on.
            prediction_output_path:  Path to a file to write out all the predictions.
        """
        super().__init__(*args, **kwargs)
        self.max_gen_toks = max_gen_toks
        self.metric = get_scrolls_metric(task_name)
        self.metric_names = metric_names
        self.num_test_examples = int(num_test_examples) if num_test_examples is not None else None
        self.predictions_output_path = Path(predictions_output_path) if predictions_output_path is not None else None
        self.task_name = task_name

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset['train']

    def _modify_input(self, doc):
        return doc

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)
        doc = self._modify_input(doc)
        return [doc]

    def _truncate_docs(self, docs):
        if self.num_test_examples is None:
            return docs
        else:
            return docs.to_list()[:self.num_test_examples]

    def validation_docs(self):
        for i, doc in enumerate(self.dataset["validation"]):
            if i == self.num_test_examples:
                break
            yield from self._process_doc(doc)

    def doc_to_text(self, doc):
        return doc['input']

    def doc_to_target(self, doc):
        return doc['output']

    def construct_requests(self, doc, ctx):
        primary_until = 'None' # TODO: set to end of text token of your model
        assert primary_until is not None, "Don't forget to set the end of text token!"
        if self.max_gen_toks is None:
            until = {'until': [primary_until]}
        else:
            until = {'until': [primary_until], 'max_gen_toks': self.max_gen_toks}

        summary = rf.greedy_until(ctx, until)
        return summary

    def process_results(self, doc, results):
        gold = [[doc['output']]]
        # write out predictions if predictions path is specified
        if self.predictions_output_path is not None:
            self.write_predictions(results[0], gold[0][0])

        # the scrolls metrics uses multiprocessing Pools to compute the Rouge score.  Using multiprocessing 'spawn' is
        # way too slow.  Default is 'fork' but parallelformers changes this to 'spawn' so we need to change
        # this back to 'fork' after.
        if mp.get_start_method != 'fork':
            mp.set_start_method('fork', force=True)
        metric_values = self.metric.compute(predictions=results, references=gold)
        return {metric_name: metric_values[metric_name] for metric_name in self.metric_names}

    def higher_is_better(self):
        return {metric_name: True for metric_name in self.metric_names}

    def aggregation(self):
        return {metric_name: mean for metric_name in self.metric_names}

    def write_predictions(self, prediction: str, gold: str):
        """Write predictions to an output file

        Args:
            prediction:  The prediction from the model.
            gold:  The gold label for the example.
        """
        line_dict = {'task': self.DATASET_NAME, 'prediction': prediction, 'gold': gold}
        with jsonlines.open(self.predictions_output_path, mode='a') as predictions_file:
            predictions_file.write(line_dict)


class GovReport(ZeroScrollsTask):
    """Summarization dataset of reports addressing various national policy issues.  Each document is paired with
    an expert-written executive summary.

    TASK STRUCTURE:
        You are given a report by a government agency. Write a one-page summary of the report.
        Report:
            {REPORT}
        Summary:
    """
    DATASET_NAME = 'gov_report'
    METRIC_NAMES = (
        'rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum',
        'rouge/geometric_mean',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(GovReport.DATASET_NAME, GovReport.METRIC_NAMES, *args, **kwargs)


class SummScreenFd(ZeroScrollsTask):
    """A summarization dataset in the domain of TV shows.  Given a transcript of a specific episode, the goal is to
    produce the episode's recap.

    TASK STRUCTURE:
        You are given a script of a TV episode. Summarize the episode in a paragraph.
        Episode Script:
            {SCRIPT}
        Summary:
    """
    DATASET_NAME = 'summ_screen_fd'
    METRIC_NAMES = (
        'rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum',
        'rouge/geometric_mean',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(SummScreenFd.DATASET_NAME, SummScreenFd.METRIC_NAMES, *args, **kwargs)


class QMSum(ZeroScrollsTask):
    """A query based summarization dataset consisting of 232 meeting transcripts from multiple domains and their
    corresponding summaries.

    TASK STRUCTURE:
        You are given a meeting transcript and a query containing a question or instruction. Answer the query in one
        or more sentences.
        Transcript:
            {TRANSCRIPT}
        Query:
            {QUERY}
        Answer:
    """
    DATASET_NAME = 'qmsum'
    METRIC_NAMES = (
        'rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum',
        'rouge/geometric_mean',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(QMSum.DATASET_NAME, QMSum.METRIC_NAMES, *args, **kwargs)


class SQuality(ZeroScrollsTask):
    """ A multiple choice question answering dataset over stories and articles.

    TASK STRUCTURE:
        You are given a story and a question. Answer the question in a paragraph.
            Story:
                {STORY}
            Question:
                {QUESTION}
            Answer:
    """
    DATASET_NAME = 'squality'
    METRIC_NAMES = (
        'rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum',
        'rouge/geometric_mean',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(SQuality.DATASET_NAME, SQuality.METRIC_NAMES, *args, **kwargs)


class Qasper(ZeroScrollsTask):
    """ Question answering over NLP papers.  Questions were written by NLP practitioners after reading only the title
    and abstract, while another set of NLP practitioners annotated the answers given the entire document.

    TASK STRUCTURE:
        You are given a scientific article and a question. Answer the question as concisely as you can, using a single
        phrase or sentence if possible. If the question cannot be answered based on the information in the article,
        write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not
        provide any explanation.
        Article:
            {ARTICLE}
        Question:
            {QUESTION}
        Answer:
    """
    DATASET_NAME = 'qasper'
    METRIC_NAMES = ('f1', )

    def __init__(self, *args, **kwargs):
        super().__init__(Qasper.DATASET_NAME, Qasper.METRIC_NAMES, *args, **kwargs)


class NarrativeQA(ZeroScrollsTask):
    """ You are given a story, which can be either a novel or a movie script, and a question. Answer the question as
    concisely as you can, using a single phrase if possible. Do not provide any explanation.

    TASK STRUCTURE:
        You are given a story, which can be either a novel or a movie script, and a question. Answer the question as
        concisely as you can, using a single phrase if possible. Do not provide any explanation.
        Story:
            {STORY}
        Question:
            {QUESTION}
        Answer:
    """
    DATASET_NAME = 'narrative_qa'
    METRIC_NAMES = ('f1', )

    def __init__(self, *args, **kwargs):
        super().__init__(NarrativeQA.DATASET_NAME, NarrativeQA.METRIC_NAMES, *args, **kwargs)


class Quality(ZeroScrollsTask):
    """ A multiple choice question answering dataset over stories and articles.

    TASK STRUCTURE:
        You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C,
        D). Choose the best answer by writing its corresponding letter (either A, B, C, or D). Do not provide any
        explanation.
        Story:
            {STORY}
        Question and Possible Answers:
            {QUESTION_AND_OPTIONS}
        Answer:
    """
    CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    DATASET_NAME = 'quality'
    METRIC_NAMES = ('accuracy', )
    _multiple_choice_pattern = re.compile(r" *\([A-D]\) *")

    @staticmethod
    def _normalize_answer(text):
        return " ".join(text.split()).strip()

    def __init__(self, *args, **kwargs):
        super().__init__(Quality.DATASET_NAME, Quality.METRIC_NAMES, *args, **kwargs)

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)

        split = doc["text"].find("\n\n", doc["text"].find("(D)"))
        choices_text = doc["text"][:split]

        doc["text"] = doc["text"][split:].strip()
        doc["choices"] = [Quality._normalize_answer(choice) for choice in re.split(
            Quality._multiple_choice_pattern, choices_text)[1:]]
        doc["gold"] = doc["output"][0]

        return [doc]

    def aggregation(self):
        return {
            "accuracy": mean,
            "accuracy_norm": mean,
        }

    def higher_is_better(self):
        return {
            "accuracy": True,
            "accuracy_norm": True,
        }

    def process_results(self, doc, results):
        gold = doc["gold"]
        gold_value = Quality.CHOICE_MAP[gold]

        pred_value = np.argmax(results)
        acc = 1.0 if pred_value == gold_value else 0.0

        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold_value else 0.0

        return {
            "accuracy": acc * 100,
            "accuracy_norm": acc_norm * 100,
        }

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls


class MuSiQue(ZeroScrollsTask):
    """Multi-hop question answering dataset where the inputs are 20 Wikipedia paragraphs and a question that requires
    multiple hops between different paragraphs.

    TASK STRUCTURE:
        You are given several paragraphs from Wikipedia and a question. Answer the question as concisely as you
        can, using a single phrase if possible. If the question cannot be answered based on the information in the
        paragraphs, write "unanswerable". Do not provide any explanation.
        Paragraphs:
            {PARAGRAPHS}
        Question:
            {QUESTION}
        Answer:
    """
    DATASET_NAME = 'musique'
    METRIC_NAMES = ('f1', )

    def __init__(self, *args, **kwargs):
        super().__init__(MuSiQue.DATASET_NAME, MuSiQue.METRIC_NAMES, *args, **kwargs)

    def _modify_input(self, doc):
        doc['question'] = '. '.join(doc['question'].split('. ')[1:])
        split_text = doc['text'].split('\n\n')
        new_input = '\n\n'.join(split_text[:-1] + [doc['question']] + [split_text[-1]])
        doc['input'] = new_input
        return doc


class SpaceDigest(ZeroScrollsTask):
    """ Given 50 hotel reviews (without their ratings) the task is to determine the percentage of positive reviews.

    TASK STRUCTURE:
        You are given a list of reviews about a specific hotel. Each review is either positive or negative. What is the
        percentage of positive reviews (e.g. 60%, 34%, etc.)? Do not provide any explanation.
        Reviews:
            {REVIEWS}
        Percentage of Positive Reviews:
    """
    DATASET_NAME = 'space_digest'
    METRIC_NAMES = ('exp_similarity', )

    def __init__(self, *args, **kwargs):
        super().__init__(SpaceDigest.DATASET_NAME, SpaceDigest.METRIC_NAMES, *args, **kwargs)

    def _modify_input(self, doc):
        start_input = '\n\n'.join(doc['input'].split('\n\n')[:-1])
        instruction_1 = 'What percent of the these reviews were positive?  '
        instruction_2 = 'For example, if 35 out of the 50 reviews are positive, then the output should be:\n\n'
        instruction_3 = 'Percentage of Positive Review:\n70%'
        instruction = instruction_1 + instruction_2 + instruction_3
        end_input = doc['input'].split('\n\n')[-1]
        doc['input'] = '\n\n'.join([start_input, instruction, end_input])
        return doc


class BookSumSort(ZeroScrollsTask):
    """Given a shuffled list of chapter summaries, the task is to reorder them according to the original order of
    summaries.

    TASK STRUCTURE:
        You are given {NUM_SUMMARIES} summaries of chapters or parts of a novel, in a shuffled order, where
        each summary is denoted by a numerical ID (e.g. Summary 1, Summary 3, etc.). Reorder the summaries
        according to the original order of chapters/parts in the novel by writing a list of length {NUM_SUMMARIES}
        of the summary IDs (e.g. if you were given 5 summaries, one possible answer could be "5, 1, 3, 4, 2"). Do
        not provide any explanation.
        Summaries:
            {SUMMARIES}
        Summary IDs in Correct Order:
    """
    DATASET_NAME = 'book_sum_sort'
    METRIC_NAMES = ('concordance_index', )

    def __init__(self, *args, **kwargs):
        super().__init__(BookSumSort.DATASET_NAME, BookSumSort.METRIC_NAMES, *args, **kwargs)

    def _modify_input(self, doc):
        example = ", ".join(map(str, list(range(1, len(doc['text'].split('\n\n'))))))
        new_input = doc['input'] + example + '\nSummary IDs in Correct Order:\n'
        doc['input'] = new_input
        return doc
