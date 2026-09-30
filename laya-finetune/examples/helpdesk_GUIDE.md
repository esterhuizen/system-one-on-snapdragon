# Labelling guide: helpdesk tickets (IT managed-services provider)

Each ticket has: `summary` (the ticket title), `request_type` (e.g. Incident, Service Request) and `helpdesk_category`
(the category a person picked when logging it; often useful, sometimes blank, generic or wrong).

Give every ticket exactly one **bucket** and one **work_type**, using ONLY the keys defined in
`taxonomy.json` (`tasks.bucket` = 12 bucket keys with their definitions; `tasks.work_type` = 6 work-type keys with definitions).

Rules:
- Pick the bucket that best describes what the IT work is about (the thing being fixed / set up / alerted on),
  not who asked or which client it is.
- Use the descriptions in taxonomy.json as the definitions. If two fit, pick the one a helpdesk manager would file it under
  when reporting what their team spent time on.
- `helpdesk_category` is evidence, not the answer: use it when the summary is vague, override it when the summary clearly
  says otherwise.
- Vague summaries (e.g. "FW:", "Issue", a person's name): decide from request_type + helpdesk_category; if still unclear,
  choose the most plausible bucket and mark confidence "low".
- work_type is about the kind of work (fix a problem / user request / automated alert / commercial / planned project /
  admin-other), independent of the bucket.
- confidence: "high" (clear), "medium" (reasonable but another bucket is plausible), "low" (guess).
