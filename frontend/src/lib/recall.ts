import { SvelteSet } from 'svelte/reactivity';

import type { PipelineResult, RecallAlert, RecallSeverity, Source } from '$lib/types/chat';

export function entityTypes(result: PipelineResult): string {
	return [...new Set(result.redaction.entities.map((entity) => entity.entity_type))].join(', ');
}

function recallSeverityFromSection(section: string): RecallSeverity {
	if (/class\s*iii\b/i.test(section)) return 'class3';
	if (/class\s*ii\b/i.test(section)) return 'class2';
	if (/class\s*i\b/i.test(section)) return 'class1';
	return 'unknown';
}

function recallLabelForSeverity(sev: RecallSeverity): string {
	if (sev === 'class1') return 'Class I recall';
	if (sev === 'class2') return 'Class II recall';
	if (sev === 'class3') return 'Class III recall';
	return 'FDA recall';
}

function isAllowedRecallUrl(raw: string): boolean {
	try {
		const u = new URL(raw);
		return u.protocol === 'https:' || u.protocol === 'http:';
	} catch {
		return false;
	}
}

/** Union retrieved + reranked so recalls dropped by reranking still surface in the UI. */
export function extractRecallAlerts(result: PipelineResult): RecallAlert[] {
	const merged = [...(result.retrieved ?? []), ...(result.reranked ?? [])];
	const seen = new SvelteSet<string>();
	const list: RecallAlert[] = [];
	for (const src of merged) {
		const meta = src.metadata ?? {};
		if (String(meta.source ?? '') !== 'openfda_recall') continue;
		if (seen.has(src.id)) continue;
		const section = String(meta.section ?? '');
		const sev = recallSeverityFromSection(section);
		const drugName = String(meta.drug_name ?? 'unknown');
		const url = String(meta.source_url ?? '');
		if (!isAllowedRecallUrl(url)) continue;
		const label = recallLabelForSeverity(sev);
		seen.add(src.id);
		list.push({
			id: src.id,
			drugName,
			severity: sev,
			label,
			sourceUrl: url
		});
	}
	list.sort((a, b) => {
		const rank = (s: RecallSeverity) =>
			s === 'class1' ? 0 : s === 'class2' ? 1 : s === 'class3' ? 2 : 3;
		const d = rank(a.severity) - rank(b.severity);
		if (d !== 0) return d;
		return a.drugName.localeCompare(b.drugName);
	});
	return list;
}

export function metadataLabel(source: Source): string {
	const metadata = source.metadata ?? {};
	const label =
		metadata.drug ??
		metadata.title ??
		metadata.source ??
		metadata.source_name ??
		metadata.url ??
		metadata.file ??
		source.id;
	return String(label);
}
