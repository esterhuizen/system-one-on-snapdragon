export const meta = {
  name: 'label-two-pass-adjudicate',
  description: 'Label batches with two independent passes, then adjudicate disagreements',
  phases: [
    { title: 'Pass A', detail: 'label each batch' },
    { title: 'Pass B', detail: 'independent second pass, different framing and order' },
    { title: 'Adjudicate', detail: 'resolve only the items where A and B disagree' },
  ],
}
// Claude Code workflow: label every batch twice, independently, then adjudicate the disagreements.
// Run it from Claude Code (ask it to "use a workflow" with this script), with args:
//   {"dir": "/abs/path/to/WORK", "batches": ["t00", "t01", ..., "v00", ...], "tasks": ["bucket", "work_type"]}
// WORK must contain GUIDE.md (labelling rules), taxonomy.json ({"tasks": {"<task>": {"<key>": "<definition>", ...}}})
// and batches/<name>.jsonl (from make_batches.py). Output: WORK/passA, WORK/passB, WORK/final (<name>.jsonl each).
// Agents write labels to files and return only small summaries, so the item text never flows back into the main session.
const DIR = args.dir, BATCHES = args.batches, TASKS = args.tasks
const FIELDS = TASKS.map(t => `"${t}": "<key from taxonomy.json tasks.${t}>"`).join(', ')
const COMMON = `Inputs (read-only): ${DIR}/GUIDE.md (labelling rules) and ${DIR}/taxonomy.json (the ONLY valid keys, under "tasks", with definitions).
Do NOT open any other file under ${DIR} except the ones named in this prompt.`
const SUMMARY = { type: 'object', properties: { batch: { type: 'string' }, n_input: { type: 'integer' }, n_written: { type: 'integer' }, note: { type: 'string' } },
  required: ['batch', 'n_input', 'n_written'] }
const ADJ = { type: 'object', properties: { batch: { type: 'string' }, n: { type: 'integer' }, disagreements: { type: 'integer' }, n_written: { type: 'integer' }, note: { type: 'string' } },
  required: ['batch', 'n', 'disagreements', 'n_written'] }
const pass = (b, which, order) => `You are labelling items. ${COMMON}
Input: ${DIR}/batches/${b}.jsonl (one item per line). Work through the items ${order}. Label every item following GUIDE.md exactly.
Write ${DIR}/${which}/${b}.jsonl: one JSON object per line {"id": "...", ${FIELDS}, "confidence": "high|medium|low"}.
Every input id must appear exactly once with valid keys; re-read your file to check, and fix any problem. Return the summary.`
const adjudicate = b => `You adjudicate two independent labelling passes. ${COMMON}
Read ${DIR}/batches/${b}.jsonl, ${DIR}/passA/${b}.jsonl and ${DIR}/passB/${b}.jsonl.
Where A and B agree on every task, keep that. Otherwise re-read the item and decide each disputed task yourself from GUIDE.md and the definitions.
Write ${DIR}/final/${b}.jsonl: one JSON object per line {"id": "...", ${FIELDS}, "confidence": "high|medium|low", "adjudicated": true|false}.
Every id exactly once, valid keys; re-read to check. Count the items where A and B disagreed on any task. Return the summary.`
const results = await pipeline(BATCHES,
  b => agent(pass(b, 'passA', 'in file order, deciding the tasks in the order listed'), { label: `A ${b}`, phase: 'Pass A', schema: SUMMARY, effort: 'medium' }),
  (a, b) => agent(pass(b, 'passB', 'from the LAST line to the FIRST, deciding the tasks in REVERSE order; someone else labels them separately, do not look for their work'),
                  { label: `B ${b}`, phase: 'Pass B', schema: SUMMARY, effort: 'medium' }),
  (x, b) => agent(adjudicate(b), { label: `adj ${b}`, phase: 'Adjudicate', schema: ADJ, effort: 'high' }).then(r => ({ batch: b, ...r })),
)
const ok = results.filter(Boolean)
return { batches_done: ok.length, missing: BATCHES.filter(b => !ok.find(r => r && r.batch === b)),
         items: ok.reduce((s, r) => s + (r.n || 0), 0), disagreements: ok.reduce((s, r) => s + (r.disagreements || 0), 0) }
