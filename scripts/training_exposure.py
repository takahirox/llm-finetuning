"""Record the actual tokenized examples delivered to the pinned MLX trainer."""

from collections import Counter
from pathlib import Path

from scripts.experiment import digest, read_jsonl


class Exposure:
    def __init__(self, dataset, configuration):
        policy = configuration['study_exposure']
        rows = read_jsonl(Path(configuration['data']) / 'train.jsonl')
        ids = policy['sample_ids']
        if len(rows) != len(ids) or len(dataset) != len(ids):
            raise ValueError('Exposure dataset count differs from frozen sample IDs.')
        by_messages = {digest(row['messages']): label for row, label in zip(rows, ids)}
        if len(by_messages) != len(ids):
            raise ValueError('Ambiguous duplicate conversations.')
        self.lookup, self.order = {}, []
        for i in range(len(dataset)):
            row = dataset[i]
            label = by_messages[digest(row['messages'])]
            tokens, offset = dataset.process(row)
            length = min(len(tokens), configuration['max_seq_length'])
            key = digest([tokens[:length], offset])
            if key in self.lookup:
                raise ValueError('Ambiguous examples after tokenization/truncation; cannot audit exposure.')
            self.lookup[key] = {'sample_id': label, 'original_tokens': len(tokens),
                                'retained_tokens': length, 'target_tokens': length - max(1, offset),
                                'truncated': length < len(tokens)}
        self.path = Path(configuration['adapter_path']) / 'exposure.jsonl'
        self.dataset_size = len(ids)
        self.path.touch(exist_ok=False)

    def observe(self, batch, lengths):
        records = []
        for tokens, (offset, length) in zip(batch, lengths):
            key = digest([tokens[:length], offset])
            if key not in self.lookup:
                raise ValueError('Observed batch differs from audited tokenization.')
            records.append(dict(self.lookup[key], ordinal=len(self.order) + len(records)))
        self.order.extend(records)
        # Persist the sequence even on interrupted runs; only complete runs are comparable.
        with self.path.open('a') as stream:
            import json
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + '\n')

    def summary(self):
        counts = Counter(row['sample_id'] for row in self.order)
        return {'samples_seen': len(self.order), 'unique_samples_seen': len(counts),
                'dataset_size': self.dataset_size, 'coverage_fraction': len(counts) / self.dataset_size,
                'sample_counts': dict(counts), 'reuse': len(self.order) - len(counts),
                'retained_tokens_seen': sum(r['retained_tokens'] for r in self.order),
                'model_input_tokens_seen': sum(r['retained_tokens'] - 1 for r in self.order),
                'discarded_tokens': sum(r['original_tokens'] - r['retained_tokens'] for r in self.order),
                'target_tokens_seen': sum(r['target_tokens'] for r in self.order),
                'truncated_presentations': sum(r['truncated'] for r in self.order),
                'order_sha256': digest(self.order)}
