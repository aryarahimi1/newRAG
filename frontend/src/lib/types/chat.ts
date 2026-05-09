export type Role = 'user' | 'assistant';
export type RecallSeverity = 'class1' | 'class2' | 'class3' | 'unknown';

export type RecallAlert = {
	id: string;
	drugName: string;
	severity: RecallSeverity;
	label: string;
	sourceUrl: string;
};

export type ChatTurn = { id: string; role: Role; content: string; recallAlerts?: RecallAlert[] };
export type StatusEntry = { id: string; text: string };

export type CorpusStats = {
	n_chunks: number;
	n_drugs: number;
	n_sources: number;
	embedding_model: string;
	collection: string;
	drugs?: string[];
};

export type MetadataValue = string | number | boolean | null | undefined;

export type Source = {
	id: string;
	text: string;
	metadata: Record<string, MetadataValue>;
	score: number;
};

export type PipelineResult = {
	question: string;
	redaction: {
		original: string;
		redacted: string;
		entities: { entity_type: string }[];
	};
	detected_drugs: { mention: string; canonical: string; rxcui: string; score: number }[];
	auto_ingest: {
		missing: { mention: string; canonical: string; rxcui: string; score: number }[];
		ingested: string[];
		added_chunks: number;
		error: string | null;
		skipped: boolean;
	};
	retrieved: Source[];
	reranked: Source[];
	generation: {
		answer: string;
		model: string;
		prompt_tokens: number | null;
		completion_tokens: number | null;
	} | null;
	timing: Record<string, number>;
	error: string | null;
	status_log?: string[];
};

export type ChatSession = {
	id: string;
	title: string;
	createdAt: number;
	updatedAt: number;
	messages: ChatTurn[];
	statusLines: StatusEntry[];
	lastResult: PipelineResult | null;
	requestError: string | null;
};
