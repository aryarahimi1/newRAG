import DOMPurify from 'isomorphic-dompurify';
import { marked } from 'marked';

marked.use({
	gfm: true,
	breaks: true
});

const ALLOWED_TAGS = [
	'p',
	'br',
	'strong',
	'em',
	'b',
	'i',
	'ul',
	'ol',
	'li',
	'h1',
	'h2',
	'h3',
	'h4',
	'h5',
	'h6',
	'blockquote',
	'code',
	'pre',
	'a',
	'span',
	'hr',
	'table',
	'thead',
	'tbody',
	'tr',
	'th',
	'td'
];

const ALLOWED_ATTR = ['href', 'title', 'target', 'rel', 'class'];

function citationSpans(html: string): string {
	return html.replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
}

/** Safe HTML for {@html}: markdown → sanitize (allow typical GFM tags). */
export function sanitizeMarkdownHtml(markdown: string): string {
	if (!markdown.trim()) return '';
	const raw = marked.parse(markdown, { async: false }) as string;
	const withCites = citationSpans(raw);
	return DOMPurify.sanitize(withCites, {
		ALLOWED_TAGS,
		ALLOWED_ATTR
	});
}
