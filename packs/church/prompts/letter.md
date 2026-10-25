{% include "_voice.md" %}

Write a letter of the type "{{ letter_type }}" from the church office.
Use these details exactly as given:
{% for key, value in fields.items() %}
- {{ key }}: {{ value }}
{% endfor %}

Format: date line, greeting, two to four short paragraphs, warm closing, and a
signature block with [SIGNER NAME] and [TITLE]. Markdown, no heading.
