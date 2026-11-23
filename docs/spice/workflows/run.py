"""Execute three local SQLite workflows with fresh mentat-Spice-4B decisions.

Inputs are public benchmark fixtures, not production business records. The model
decides; the Python controller retrieves, persists, queues and extracts text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Workflows:
    def __init__(self, model, output, fixtures):
        self.model, self.output, self.fixtures = model, output, fixtures
        self.db = sqlite3.connect(output / 'workflow.sqlite')
        self.db.row_factory = sqlite3.Row
        self.started = time.perf_counter()
        self.trace = []
        self.db.executescript('''
            CREATE TABLE source_records(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE tickets(id TEXT PRIMARY KEY, queue TEXT, status TEXT, priority TEXT, decision TEXT);
            CREATE TABLE invoices(id TEXT PRIMARY KEY, disposition TEXT, status TEXT, decision TEXT);
            CREATE TABLE review_queue(kind TEXT, object_id TEXT, reason TEXT, PRIMARY KEY(kind,object_id));
            CREATE TABLE documents(id INTEGER PRIMARY KEY, source_id TEXT, body TEXT);
            CREATE VIRTUAL TABLE search USING fts5(body, content='documents', content_rowid='id');
            CREATE TABLE summaries(id TEXT PRIMARY KEY, text TEXT, status TEXT, verification TEXT);
            CREATE TABLE events(seq INTEGER PRIMARY KEY, workflow TEXT, step TEXT, elapsed_ms REAL, payload TEXT);
        ''')

    def event(self, workflow, step, **payload):
        event = {'seq': len(self.trace) + 1, 'workflow': workflow, 'step': step,
                 'elapsed_ms': round((time.perf_counter() - self.started) * 1000, 3), **payload}
        self.trace.append(event)
        self.db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                        (event['seq'], workflow, step, event['elapsed_ms'], json.dumps(payload)))
        self.db.commit()
        write(self.output / 'events.json', self.trace)
        print(json.dumps({'workflow': workflow, 'step': step}, ensure_ascii=False), flush=True)

    def infer(self, workflow, request, **metadata):
        response = self.model.predict(request)
        self.event(workflow, 'model_decision', request=request, response=response, **metadata)
        return response

    def ingest(self, fixture):
        self.db.execute('INSERT INTO source_records VALUES(?,?)',
                        (fixture['case_id'], json.dumps(fixture['request'], ensure_ascii=False)))
        self.db.commit()
        return json.loads(self.db.execute('SELECT payload FROM source_records WHERE id=?',
                                        (fixture['case_id'],)).fetchone()[0])

    def customer(self):
        fixture = self.fixtures['customer']
        request = self.ingest(fixture)
        self.event('customer', 'context_loaded', case_id=fixture['case_id'], state=request['state'],
                   source='SQLite source_records', queue_count_before=0)
        response = self.infer('customer', request)
        answers = response['answers']
        action, category = answers['action']['choice'], answers['category']['choice']
        needs_human = answers['needs_human']['noul'] >= .5
        abstained = any(a.get('abstained', False) for a in answers.values())
        queue = f'{category}_support' if not abstained else 'manual_triage'
        status = 'awaiting_human' if needs_human or action == 'escalate_to_human' or abstained else 'triaged'
        # The controller translates the model's rubric score into a demo queue priority.
        urgency = answers['urgency']['score']
        priority = 'critical' if urgency >= 2.5 else 'high' if urgency >= 1.5 else 'routine'
        self.db.execute('INSERT INTO tickets VALUES(?,?,?,?,?)',
                        (fixture['case_id'], queue, status, priority, json.dumps(response)))
        if status == 'awaiting_human':
            self.db.execute('INSERT INTO review_queue VALUES(?,?,?)',
                            ('ticket', fixture['case_id'], 'Human review selected by the model or required by uncertainty policy.'))
        self.db.commit()
        final = dict(self.db.execute('SELECT id,queue,status,priority FROM tickets WHERE id=?',
                                     (fixture['case_id'],)).fetchone())
        count = self.db.execute("SELECT count(*) FROM review_queue WHERE kind='ticket'").fetchone()[0]
        self.event('customer', 'ticket_created', database_row=final, review_queue_count=count,
                   executor='Python controller: SQLite transaction')
        result = {'workflow': 'customer', 'outcome': final, 'review_queue_count': count,
                  'verified': status == 'awaiting_human' and category == 'technical' and count == 1,
                  'resolved_customer_issue': False, 'input_source': fixture['case_id']}
        self.event('customer', 'postcondition_checked', result=result)
        return result

    def invoice(self):
        cases = []
        for key in ['invoice_mismatch', 'invoice_matching']:
            fixture = self.fixtures[key]
            request = self.ingest(fixture)
            state = request['state']
            self.event('invoice', 'records_loaded', case=key, case_id=fixture['case_id'],
                       state=state, source='SQLite source_records')
            response = self.infer('invoice', request, case=key)
            answer = response['answers']['disposition']
            disposition = answer['choice']
            quantities_match = (state['invoice']['lines'][0]['qty'] == state['delivery']['received_qty']
                                == state['purchase_order']['lines'][0]['qty'])
            totals_match = state['invoice']['total_usd'] == state['purchase_order']['total_usd']
            # Explicit accounting guard: model approval alone never releases mismatched items.
            approved = (disposition == 'approve' and not answer.get('abstained')
                        and quantities_match and totals_match and state['delivery']['condition'] == 'accepted')
            status = 'approved_for_payment' if approved else 'held_for_review'
            invoice_id = state['invoice']['id']
            self.db.execute('INSERT INTO invoices VALUES(?,?,?,?)',
                            (invoice_id, disposition, status, json.dumps(response)))
            if not approved:
                self.db.execute('INSERT INTO review_queue VALUES(?,?,?)',
                                ('invoice', invoice_id, f'Model disposition={disposition}; quantities_match={quantities_match}; totals_match={totals_match}.'))
            self.db.commit()
            row = dict(self.db.execute('SELECT id,disposition,status FROM invoices WHERE id=?', (invoice_id,)).fetchone())
            self.event('invoice', 'review_status_written', case=key, database_row=row,
                       quantities_match=quantities_match, totals_match=totals_match,
                       executor='Python controller: model disposition + reconciliation guard')
            cases.append({'case': key, 'source_case_id': fixture['case_id'], 'state': state,
                          'response': response, 'outcome': row})
        result = {'workflow': 'invoice', 'cases': cases,
                  'verified': cases[0]['outcome']['status'] == 'held_for_review'
                              and cases[1]['outcome']['status'] == 'approved_for_payment',
                  'external_payments_executed': 0}
        self.event('invoice', 'postcondition_checked', result=result)
        return result

    def grounded_summary(self):
        fixture = self.fixtures['rag']
        original = fixture['request']
        for i, doc in enumerate(self.fixtures['news_corpus'], 1):
            self.db.execute('INSERT INTO documents VALUES(?,?,?)', (i, doc['source_id'], doc['body']))
        self.db.execute("INSERT INTO search(search) VALUES('rebuild')")
        self.db.commit()
        query = self.fixtures['news_query']
        found = self.db.execute('''SELECT documents.source_id,documents.body,bm25(search) AS rank
                    FROM search JOIN documents ON documents.id=search.rowid
                    WHERE search MATCH ? ORDER BY rank LIMIT 3''', (query,)).fetchall()
        if not found:
            raise RuntimeError('No source retrieved; do not fabricate a publishing workflow.')
        source = dict(found[0])
        self.event('rag', 'source_retrieved', query=query, corpus_documents=len(self.fixtures['news_corpus']),
                   results=[dict(r) for r in found], executor='SQLite FTS5 / BM25')
        prefix = 'Summarize the following news within 38 words:\n'
        source_prompt = prefix + source['body'] + '\n\noutput:'
        request = {'state': {'prompt': source_prompt, 'response': original['state']['response']},
                   'questions': original['questions']}
        self.db.execute('INSERT INTO summaries VALUES(?,?,?,?)', ('draft', request['state']['response'], 'pending', None))
        self.db.commit()
        first = self.infer('rag', request, revision='original_draft')
        q = first['answers']['q']
        unsupported = q['noul'] >= .5 or q.get('abstained', False)
        self.db.execute('UPDATE summaries SET status=?,verification=? WHERE id=?',
                        ('blocked' if unsupported else 'published', json.dumps(first), 'draft'))
        self.db.commit()
        self.event('rag', 'draft_gate_written', status='blocked' if unsupported else 'published',
                   unsupported_probability=q['noul'], database_row_id='draft')
        if unsupported:
            # This is a deterministic, extractive fallback, not a model-generated rewrite.
            sentences = re.split(r'(?<=[.!?])\s+', source['body'])
            selected = []
            for sentence in sentences:
                if len(' '.join([*selected, sentence]).split()) > 38:
                    break
                selected.append(sentence)
            replacement = ' '.join(selected)
            if not replacement:
                raise RuntimeError('No full sentence fits the extractive fallback; leave the draft blocked.')
            self.event('rag', 'extractive_fallback_created', text=replacement, words=len(replacement.split()),
                       executor='Python controller: leading complete source sentences, at most 38 words',
                       generated_by_model=False)
            revised_request = {'state': {'prompt': source_prompt, 'response': replacement},
                               'questions': original['questions']}
            second = self.infer('rag', revised_request, revision='extractive_fallback')
            answer = second['answers']['q']
            accepted = answer['noul'] < .5 and not answer.get('abstained', False)
            status = 'published' if accepted else 'held_for_review'
            self.db.execute('INSERT INTO summaries VALUES(?,?,?,?)',
                            ('replacement', replacement, status, json.dumps(second)))
            self.db.commit()
            self.event('rag', 'replacement_status_written', database_row_id='replacement', status=status,
                       text=replacement, unsupported_probability=answer['noul'])
        else:
            second = None
        rows = [dict(r) for r in self.db.execute('SELECT id,text,status FROM summaries ORDER BY rowid')]
        result = {'workflow': 'rag', 'source_case_id': fixture['case_id'], 'retrieved_source_id': source['source_id'],
                  'original_source_retrieved': source['source_id'] == fixture['case_id'],
                  'first_response': first, 'second_response': second, 'outcomes': rows,
                  'verified': ((unsupported and len(rows) == 2 and rows[1]['status'] == 'published')
                               or (not unsupported and len(rows) == 1 and rows[0]['status'] == 'published'))
                              and source['source_id'] == fixture['case_id'],
                  'rewrite_method': 'Deterministic source extraction; mentat verifies both drafts.',
                  'publication_target': 'Local SQLite summaries table only'}
        self.event('rag', 'postcondition_checked', result=result)
        return result

    def run(self):
        try:
            results = [self.customer(), self.invoice(), self.grounded_summary()]
            integrity = self.db.execute('PRAGMA integrity_check').fetchone()[0]
            final = {'model': 'mentat-Spice-4B', 'mode': 'fresh_native_inference_and_local_SQLite_execution',
                     'environment': 'Local sandbox; public benchmark fixtures, not production business data.',
                     'results': results, 'passed': all(r['verified'] for r in results) and integrity == 'ok',
                     'database_integrity': integrity, 'events': len(self.trace),
                     'external_service_calls_or_money_movements': 0,
                     'fixtures_sha256': sha(HERE/'fixtures.json')}
            write(self.output/'receipt.json', final)
            for workflow in ['customer','invoice','rag']:
                write(self.output/f'{workflow}.json', {'workflow': workflow,
                      'events': [e for e in self.trace if e['workflow'] == workflow],
                      'result': next(r for r in results if r['workflow'] == workflow),
                      'fresh_inference': True, 'local_SQLite_execution': True,
                      'environment': final['environment'], 'model': final['model']})
            print(json.dumps({'passed': final['passed'], 'events': len(self.trace), 'output': str(self.output)}), flush=True)
            return final
        finally:
            self.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', help='Optional local pinned backbone')
    parser.add_argument('--adapter-directory', help='Optional local adapter directory; requires --base-model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path, help='A new directory; existing runs are never overwritten')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from mentat_spice import DecisionModel
    if args.adapter_directory:
        if not args.base_model:
            parser.error('--adapter-directory requires --base-model')
        model = DecisionModel(args.base_model, args.adapter_directory, device=args.device)
    else:
        model = DecisionModel.from_pretrained(base_model=args.base_model, device=args.device)
    runner = Workflows(model, args.output, json.loads((HERE/'fixtures.json').read_text()))
    result = runner.run()
    if not result['passed']:
        raise SystemExit('Workflow did not reach every intended postcondition; the actual results were retained.')


if __name__ == '__main__':
    main()
