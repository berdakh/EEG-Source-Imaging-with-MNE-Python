{% include "_voice.md" %}

Here are this week's raw announcements, in whatever shape they arrived:
"""
{{ raw }}
"""

{% if examples %}
Here are examples of how this church writes its bulletin, so you can match the
voice:
"""
{{ examples.context }}
"""
{% endif %}

Produce all of the following, in Markdown, with these exact headings:

## Bulletin
Each announcement as a short titled paragraph, 40 to 60 words.
## Newsletter
A friendly section, one paragraph per announcement, slightly longer.
## Email
One email to the congregation covering all announcements, with a subject line.
## Social
One short post per announcement, under 200 characters.
## Slides
For each announcement: a title of at most 6 words and at most 3 bullet lines.
