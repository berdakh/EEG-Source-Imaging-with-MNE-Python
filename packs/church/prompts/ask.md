{% include "_voice.md" %}

Answer the question using only the passages below. Cite passages by their
number in square brackets, like [2]. If the passages do not contain the answer,
say: "I could not find this in the office documents." Do not guess.

Passages:
{{ retrieve.context }}

Question: {{ question }}
