import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { LangChainTracer } from '@langchain/core/tracers/tracer_langchain';
import { Logger } from '@n8n/backend-common';
import { Service } from '@n8n/di';
import { AiAssistantClient, AiAssistantSDK } from '@n8n_io/ai-assistant-sdk';
import assert from 'assert';
import { Client as TracingClient } from 'langsmith';
import type { IUser, INodeTypeDescription, ITelemetryTrackProperties } from 'n8n-workflow';

import { LLMServiceError } from '@/errors';
import { anthropicClaudeSonnet45 } from '@/llm-config';
import { SessionManagerService } from '@/session-manager.service';
import { getProxyAgent } from '@/utils/http-proxy-agent';
import {
	BuilderFeatureFlags,
	WorkflowBuilderAgent,
	type ChatPayload,
} from '@/workflow-builder-agent';

type OnCreditsUpdated = (userId: string, creditsQuota: number, creditsClaimed: number) => void;

type OnTelemetryEvent = (event: string, properties: ITelemetryTrackProperties) => void;

@Service()
export class AiWorkflowBuilderService {
	private readonly parsedNodeTypes: INodeTypeDescription[];
	private sessionManager: SessionManagerService;

	constructor(
		parsedNodeTypes: INodeTypeDescription[],
		private readonly client?: AiAssistantClient,
		private readonly logger?: Logger,
		private readonly instanceId?: string,
		private readonly instanceUrl?: string,
		private readonly n8nVersion?: string,
		private readonly onCreditsUpdated?: OnCreditsUpdated,
		private readonly onTelemetryEvent?: OnTelemetryEvent,
	) {
		this.parsedNodeTypes = this.filterNodeTypes(parsedNodeTypes);
		this.sessionManager = new SessionManagerService(this.parsedNodeTypes, logger);
	}

	private static async getAnthropicClaudeModel({
		baseUrl,
		authHeaders = {},
		apiKey = '-',
	}: {
		baseUrl?: string;
		authHeaders?: Record<string, string>;
		apiKey?: string;
	} = {}): Promise<BaseChatModel> {
		// 从环境变量读取模型类型和配置
		const modelProvider = process.env.N8N_AI_BUILDER_PROVIDER || 'anthropic';
		const modelName = process.env.N8N_AI_BUILDER_MODEL || 'claude-sonnet-4-5';
		const customBaseUrl = process.env.N8N_AI_BUILDER_BASE_URL || baseUrl;
		const customApiKey = process.env.N8N_AI_BUILDER_API_KEY || apiKey;

		// 如果使用 OpenAI 兼容的 API（包括 DeepSeek）
		if (modelProvider === 'openai' || modelProvider === 'deepseek') {
			const { ChatOpenAI } = await import('@langchain/openai');
			const defaultBaseUrl =
				modelProvider === 'deepseek' ? 'https://api.deepseek.com/v1' : 'https://api.openai.com/v1';

			// 为 DeepSeek 创建自定义 fetch 函数，拦截并移除 response_format
			const fetchOptions: any = {
				dispatcher: getProxyAgent(customBaseUrl || defaultBaseUrl),
			};

			if (modelProvider === 'deepseek') {
				// 保存原始的 fetch 函数
				const originalFetch = global.fetch;

				// 创建自定义 fetch，拦截请求并移除 response_format
				fetchOptions.fetch = async (url: string | URL, init?: RequestInit): Promise<Response> => {
					const urlString = typeof url === 'string' ? url : url.toString();

					// 如果是 DeepSeek API 请求，拦截并修改请求体
					if (urlString.includes('deepseek.com') && init?.body) {
						try {
							// 解析请求体
							let body: any;
							if (typeof init.body === 'string') {
								body = JSON.parse(init.body);
							} else if (init.body instanceof FormData) {
								// FormData 不需要处理
								return originalFetch(url, init);
							} else {
								body = init.body;
							}

							// 移除 response_format 参数
							if (body && typeof body === 'object') {
								if ('response_format' in body) {
									delete body.response_format;
								}
								// 重新序列化请求体
								init.body = JSON.stringify(body);
							}
						} catch {
							// 如果解析失败，继续使用原始请求（忽略错误）
						}
					}

					return originalFetch(url, init);
				};
			}

			// 对于 DeepSeek，确保不设置 response_format
			// 通过 modelKwargs 显式设置为 undefined（如果 LangChain 支持）
			const modelKwargs: Record<string, unknown> | undefined =
				modelProvider === 'deepseek' ? {} : undefined;

			const chatModel = new ChatOpenAI({
				model: modelName, // 例如: 'deepseek-chat' 或 'gpt-4'
				apiKey: customApiKey,
				temperature: 0,
				maxTokens: -1,
				modelKwargs, // 确保不传递 response_format
				configuration: {
					baseURL: customBaseUrl || defaultBaseUrl,
					defaultHeaders: authHeaders,
					fetchOptions,
				},
			});

			// DeepSeek 不支持某些 response_format 类型，需要拦截 bindTools
			if (modelProvider === 'deepseek') {
				const originalBindTools = chatModel.bindTools.bind(chatModel);
				chatModel.bindTools = function (tools, kwargs) {
					// 调用原始的 bindTools，但传入 kwargs 时排除 responseFormat
					const cleanKwargs = kwargs ? { ...kwargs } : {};
					if ('responseFormat' in cleanKwargs) {
						delete cleanKwargs.responseFormat;
					}
					const boundModel = originalBindTools(
						tools,
						Object.keys(cleanKwargs).length > 0 ? cleanKwargs : undefined,
					);

					// 如果 boundModel 有 responseFormat 属性，移除它
					if (boundModel && 'responseFormat' in boundModel) {
						delete (boundModel as any).responseFormat;
					}

					// 拦截底层 HTTP 请求，移除 response_format 参数
					// 通过修改 boundModel 的内部配置
					if (boundModel && (boundModel as any).lc_kwargs) {
						const lcKwargs = (boundModel as any).lc_kwargs;
						if (lcKwargs && lcKwargs.responseFormat) {
							delete lcKwargs.responseFormat;
						}
					}

					// 拦截 invoke 调用，确保不传递 response_format
					if (boundModel && typeof boundModel.invoke === 'function') {
						const originalInvoke = boundModel.invoke.bind(boundModel);
						boundModel.invoke = async function (input, options) {
							// 如果 input 中有 response_format，移除它
							if (input && typeof input === 'object' && 'response_format' in input) {
								const { response_format, ...restInput } = input as any;
								return originalInvoke(restInput, options);
							}
							return originalInvoke(input, options);
						};
					}

					// 拦截底层 HTTP 请求配置
					// 通过修改 configuration 来移除 response_format
					if (boundModel && (boundModel as any).configuration) {
						const config = (boundModel as any).configuration;
						if (config && config.defaultParams && config.defaultParams.response_format) {
							delete config.defaultParams.response_format;
						}
					}

					return boundModel;
				};
			}

			return chatModel;
		}

		// 默认使用 Anthropic
		return await anthropicClaudeSonnet45({
			baseUrl: customBaseUrl,
			apiKey: customApiKey,
			headers: {
				...authHeaders,
				// eslint-disable-next-line @typescript-eslint/naming-convention
				'anthropic-beta': 'prompt-caching-2024-07-31',
			},
		});
	}

	private async getApiProxyAuthHeaders(user: IUser, userMessageId: string) {
		assert(this.client);

		const authResponse = await this.client.getBuilderApiProxyToken(user, { userMessageId });
		const authHeaders = {
			// eslint-disable-next-line @typescript-eslint/naming-convention
			Authorization: `${authResponse.tokenType} ${authResponse.accessToken}`,
		};

		return authHeaders;
	}

	private async setupModels(
		user: IUser,
		userMessageId: string,
	): Promise<{
		anthropicClaude: BaseChatModel;
		tracingClient?: TracingClient;
		// eslint-disable-next-line @typescript-eslint/naming-convention
		authHeaders?: { Authorization: string };
	}> {
		try {
			// If client is provided, use it for API proxy
			if (this.client) {
				const authHeaders = await this.getApiProxyAuthHeaders(user, userMessageId);

				// Extract baseUrl from client configuration
				const baseUrl = this.client.getApiProxyBaseUrl();

				const anthropicClaude = await AiWorkflowBuilderService.getAnthropicClaudeModel({
					baseUrl: baseUrl + '/anthropic',
					authHeaders,
				});

				const tracingClient = new TracingClient({
					apiKey: '-',
					apiUrl: baseUrl + '/langsmith',
					autoBatchTracing: false,
					traceBatchConcurrency: 1,
					fetchOptions: {
						headers: {
							...authHeaders,
						},
					},
				});

				return { tracingClient, anthropicClaude, authHeaders };
			}

			// If base URL is not set, use environment variables
			// 优先使用 N8N_AI_BUILDER_API_KEY（用于 DeepSeek/OpenAI），否则使用 N8N_AI_ANTHROPIC_KEY（用于 Anthropic）
			const apiKey = process.env.N8N_AI_BUILDER_API_KEY || process.env.N8N_AI_ANTHROPIC_KEY || '';
			const anthropicClaude = await AiWorkflowBuilderService.getAnthropicClaudeModel({
				apiKey,
			});

			return { anthropicClaude };
		} catch (error) {
			const errorMessage = error instanceof Error ? `: ${error.message}` : '';
			const llmError = new LLMServiceError(`Failed to connect to LLM Provider${errorMessage}`, {
				cause: error,
				tags: {
					hasClient: !!this.client,
					hasUser: !!user,
				},
			});
			throw llmError;
		}
	}

	private filterNodeTypes(nodeTypes: INodeTypeDescription[]): INodeTypeDescription[] {
		// These types are ignored because they tend to cause issues when generating workflows
		const ignoredTypes = new Set([
			'@n8n/n8n-nodes-langchain.toolVectorStore',
			'@n8n/n8n-nodes-langchain.documentGithubLoader',
			'@n8n/n8n-nodes-langchain.code',
		]);

		const visibleNodeTypes = nodeTypes.filter(
			(nodeType) =>
				// We filter out hidden nodes, except for the Data Table node which has custom hiding logic
				// See more details in DataTable.node.ts#L29
				!ignoredTypes.has(nodeType.name) &&
				(nodeType.hidden !== true || nodeType.name === 'n8n-nodes-base.dataTable'),
		);

		return visibleNodeTypes.map((nodeType) => {
			// If the node type is a tool, we need to find the corresponding non-tool node type
			// and merge the two node types to get the full node type description.
			const isTool = nodeType.name.endsWith('Tool');
			if (!isTool) return nodeType;

			const nonToolNode = nodeTypes.find((nt) => nt.name === nodeType.name.replace('Tool', ''));
			if (!nonToolNode) return nodeType;

			return {
				...nonToolNode,
				...nodeType,
			};
		});
	}

	private async getAgent(user: IUser, userMessageId: string, featureFlags?: BuilderFeatureFlags) {
		const { anthropicClaude, tracingClient, authHeaders } = await this.setupModels(
			user,
			userMessageId,
		);

		const agent = new WorkflowBuilderAgent({
			parsedNodeTypes: this.parsedNodeTypes,
			// We use Sonnet both for simple and complex tasks
			llmSimpleTask: anthropicClaude,
			llmComplexTask: anthropicClaude,
			logger: this.logger,
			checkpointer: this.sessionManager.getCheckpointer(),
			tracer: tracingClient
				? new LangChainTracer({ client: tracingClient, projectName: 'n8n-workflow-builder' })
				: undefined,
			instanceUrl: this.instanceUrl,
			onGenerationSuccess: async () => {
				await this.onGenerationSuccess(user, authHeaders);
			},
			runMetadata: {
				n8nVersion: this.n8nVersion,
				featureFlags: featureFlags ?? {},
			},
		});

		return agent;
	}

	private async onGenerationSuccess(
		user?: IUser,
		// eslint-disable-next-line @typescript-eslint/naming-convention
		authHeaders?: { Authorization: string },
	): Promise<void> {
		try {
			if (this.client) {
				assert(authHeaders, 'Auth headers must be set when AI Assistant Service client is used');
				assert(user);
				const creditsInfo = await this.client.markBuilderSuccess(user, authHeaders);

				// Call the callback with the credits info from the response
				if (this.onCreditsUpdated && user.id && creditsInfo) {
					this.onCreditsUpdated(user.id, creditsInfo.creditsQuota, creditsInfo.creditsClaimed);
				}
			}
		} catch (error: unknown) {
			if (error instanceof Error) {
				this.logger?.error(`Unable to mark generation success ${error.message}`, { error });
			}
		}
	}

	async *chat(payload: ChatPayload, user: IUser, abortSignal?: AbortSignal) {
		const agent = await this.getAgent(user, payload.id, payload.featureFlags);
		const userId = user?.id?.toString();
		const workflowId = payload.workflowContext?.currentWorkflow?.id;

		for await (const output of agent.chat(payload, userId, abortSignal)) {
			yield output;
		}

		// After the stream completes, track telemetry
		if (this.onTelemetryEvent && userId) {
			try {
				await this.trackBuilderReplyTelemetry(agent, workflowId, userId, payload.id);
			} catch (error) {
				this.logger?.error('Failed to track builder reply telemetry', { error });
			}
		}
	}

	private async trackBuilderReplyTelemetry(
		agent: WorkflowBuilderAgent,
		workflowId: string | undefined,
		userId: string,
		userMessageId: string,
	): Promise<void> {
		if (!this.onTelemetryEvent) return;

		const state = await agent.getState(workflowId, userId);
		const threadId = SessionManagerService.generateThreadId(workflowId, userId);

		// extract the last message that was sent to the user for telemetry
		const lastAiMessage = state.values.messages.findLast(
			(m: BaseMessage): m is AIMessage => m instanceof AIMessage,
		);
		const messageAi =
			typeof lastAiMessage?.content === 'string'
				? lastAiMessage.content
				: JSON.stringify(lastAiMessage?.content ?? '');

		const toolMessages = state.values.messages.filter(
			(m: BaseMessage): m is ToolMessage => m instanceof ToolMessage,
		);
		const toolsCalled = [
			...new Set(
				toolMessages
					.map((m: ToolMessage) => m.name)
					.filter((name: string | undefined): name is string => name !== undefined),
			),
		];

		// Build telemetry properties
		const properties: ITelemetryTrackProperties = {
			user_id: userId,
			instance_id: this.instanceId,
			workflow_id: workflowId,
			sequence_id: threadId,
			message_ai: messageAi,
			tools_called: toolsCalled,
			techniques_categories: state.values.techniqueCategories,
			validations: state.values.validationHistory,
			// Only include templates_selected when templates were actually used
			...(state.values.templateIds.length > 0 && {
				templates_selected: state.values.templateIds,
			}),
			user_message_id: userMessageId,
		};

		this.onTelemetryEvent('Builder replied to user message', properties);
	}

	async getSessions(workflowId: string | undefined, user?: IUser) {
		const userId = user?.id?.toString();
		return await this.sessionManager.getSessions(workflowId, userId);
	}

	async getBuilderInstanceCredits(
		user: IUser,
	): Promise<AiAssistantSDK.BuilderInstanceCreditsResponse> {
		if (this.client) {
			return await this.client.getBuilderInstanceCredits(user);
		}

		// if using env variables directly instead of ai proxy service
		return {
			creditsQuota: -1,
			creditsClaimed: 0,
		};
	}

	/**
	 * Truncate all messages including and after the message with the specified messageId
	 * Used when restoring to a previous version
	 */
	async truncateMessagesAfter(
		workflowId: string,
		user: IUser,
		messageId: string,
	): Promise<boolean> {
		return await this.sessionManager.truncateMessagesAfter(workflowId, user.id, messageId);
	}
}
