# Judge calibration: `gemma3:27b` on the 30 fact rows

| Control | Expected verdict | Correct | Partial | Incorrect | Judge accuracy |
|---|---|---|---|---|---|
| positive | correct | 29 | 1 | 0 | 29/30 (97%) |
| negative | incorrect | 0 | 1 | 29 | 29/30 (97%) |

## Rows the judge got wrong

| Control | Id | Verdict | Judge's rationale |
|---|---|---|---|
| positive | G032 | partial | The response accurately states three of the six consumer rights defined in the Act, but omits the rights to be heard, to seek redressal, and to consumer awareness. While not incorrect, it is incomplete. |
| negative | G056 | partial | The response provides the definition of 'deficiency' but does not answer the core question of whether a consumer *can* sue for deficiency in service regarding marine insurance. It states a relevant part of the Act, but fails to state the key point from the gold answer that a consumer must prove they *are* a consumer and that a deficiency occurred. |
