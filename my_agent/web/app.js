"use strict";

const bootstrap = window.__MY_AGENT__;

const elements = {
  body: document.body,
  workspacePath: document.querySelector("#workspace-path"),
  sessionList: document.querySelector("#session-list"),
  sessionTitle: document.querySelector("#session-title"),
  connectionStatus: document.querySelector("#connection-status"),
  conversation: document.querySelector("#conversation"),
  emptyState: document.querySelector("#empty-state"),
  newSession: document.querySelector("#new-session"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  inspectorToggle: document.querySelector("#inspector-toggle"),
  inspectorClose: document.querySelector("#inspector-close"),
  backdrop: document.querySelector("#backdrop"),
  modeSelector: document.querySelector("#mode-selector"),
  modelSelect: document.querySelector("#model-select"),
  composer: document.querySelector("#composer"),
  promptInput: document.querySelector("#prompt-input"),
  sendPrompt: document.querySelector("#send-prompt"),
  stopTurn: document.querySelector("#stop-turn"),
  approvalPanel: document.querySelector("#approval-panel"),
  approvalTitle: document.querySelector("#approval-title"),
  approvalDetails: document.querySelector("#approval-details"),
  planPanel: document.querySelector("#plan-panel"),
  executePlan: document.querySelector("#execute-plan"),
  revisePlan: document.querySelector("#revise-plan"),
  recoveryPanel: document.querySelector("#recovery-panel"),
  recoveryError: document.querySelector("#recovery-error"),
  retryTurn: document.querySelector("#retry-turn"),
  abortTurn: document.querySelector("#abort-turn"),
  credentialPanel: document.querySelector("#credential-panel"),
  credentialTitle: document.querySelector("#credential-title"),
  credentialError: document.querySelector("#credential-error"),
  credentialPath: document.querySelector("#credential-path"),
  credentialKey: document.querySelector("#credential-key"),
  credentialSave: document.querySelector("#credential-save"),
  credentialAbort: document.querySelector("#credential-abort"),
  contextPercent: document.querySelector("#context-percent"),
  contextTokens: document.querySelector("#context-tokens"),
  usageFill: document.querySelector("#usage-fill"),
  contextBreakdown: document.querySelector("#context-breakdown"),
  contextMeta: document.querySelector("#context-meta"),
  activityList: document.querySelector("#activity-list"),
  clearActivity: document.querySelector("#clear-activity"),
  toast: document.querySelector("#toast"),
};

const state = {
  sessions: [],
  models: [],
  modes: [],
  selectedSession: null,
  busySessions: new Set(),
  attentionSessions: new Set(),
  lastSequence: new Map(),
  liveTurns: new Map(),
  activities: [],
  currentAssistant: null,
  pendingApproval: null,
  credentialRequest: null,
  credentialsPath: "",
  credentialSaving: false,
  stopped: false,
};

let toastTimer = null;

function requestID() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `request-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function rpc(method, params = {}) {
  const response = await fetch("/api/rpc", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-My-Agent-Token": bootstrap.token,
    },
    body: JSON.stringify({
      channel: "request",
      rpc_version: bootstrap.rpcVersion,
      request_id: requestID(),
      method,
      params,
    }),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const envelope = await response.json();
  if (!envelope.ok) {
    const error = new Error(envelope.error?.message || "RPC failed");
    error.code = envelope.error?.code;
    error.data = envelope.error?.data;
    throw error;
  }
  return envelope.result || {};
}

function setConnection(label, connectionState) {
  elements.connectionStatus.textContent = label;
  elements.connectionStatus.dataset.state = connectionState;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 4500);
}

function shortSessionID(sessionID) {
  const parts = sessionID.split("-");
  return parts.length > 2 ? parts.slice(-2).join("-") : sessionID;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatTokens(value) {
  const count = Number(value || 0);
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return String(count);
}

function selectedSessionID() {
  return state.selectedSession?.session_id || null;
}

function sessionSummary(sessionID) {
  return state.sessions.find((item) => item.session_id === sessionID);
}

function updateSessionSummary(sessionID, updates = {}) {
  const item = sessionSummary(sessionID);
  if (item) Object.assign(item, updates);
  renderSessionList();
}

function renderSessionList() {
  const fragment = document.createDocumentFragment();
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.dataset.sessionId = session.session_id;
    if (session.session_id === selectedSessionID()) {
      button.setAttribute("aria-current", "page");
    }

    const name = document.createElement("span");
    name.className = "session-name";
    name.textContent = shortSessionID(session.session_id);

    const indicator = document.createElement("span");
    indicator.className = "session-indicator";
    if (state.busySessions.has(session.session_id)) {
      indicator.dataset.state = "busy";
      indicator.title = "运行中";
    } else if (state.attentionSessions.has(session.session_id)) {
      indicator.dataset.state = "attention";
      indicator.title = "需要处理";
    }

    const meta = document.createElement("span");
    meta.className = "session-meta";
    meta.textContent = `${session.message_count || 0} 条消息 · ${formatTime(session.updated_at)}`;

    button.append(name, indicator, meta);
    button.addEventListener("click", () => selectSession(session.session_id));
    fragment.append(button);
  }
  elements.sessionList.replaceChildren(fragment);
}

function renderModelCatalog() {
  const fragment = document.createDocumentFragment();
  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.ref;
    option.textContent = model.description ? `${model.ref} · ${model.description}` : model.ref;
    fragment.append(option);
  }
  elements.modelSelect.replaceChildren(fragment);
}

function renderModeCatalog() {
  const fragment = document.createDocumentFragment();
  for (const mode of state.modes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mode-button";
    button.textContent = mode.mode_id;
    button.title = mode.description || mode.mode_id;
    button.dataset.modeId = mode.mode_id;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => switchMode(mode.mode_id));
    fragment.append(button);
  }
  elements.modeSelector.replaceChildren(fragment);
}

function updateControls() {
  const session = state.selectedSession;
  const sessionID = selectedSessionID();
  const busy = sessionID ? state.busySessions.has(sessionID) : false;
  const pending = Boolean(session?.pending_turn_id);
  const credentialNeeded = Boolean(
    pending && state.credentialRequest?.sessionID === sessionID,
  );
  const enabled = Boolean(session);

  elements.sessionTitle.textContent = sessionID || "未选择会话";
  elements.promptInput.disabled = !enabled || busy || pending;
  elements.sendPrompt.disabled = !enabled || busy || pending;
  elements.stopTurn.hidden = !busy;
  elements.modelSelect.disabled = !enabled || busy || pending;

  if (session?.provider_id && session?.model_id) {
    elements.modelSelect.value = `${session.provider_id}/${session.model_id}`;
  }
  for (const button of elements.modeSelector.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.modeId === session?.mode_id));
    button.disabled = !enabled || busy || pending;
  }

  const proposed = session?.proposed_plan_status === "proposed";
  elements.planPanel.hidden = !proposed || busy;
  elements.recoveryPanel.hidden = !pending || busy || credentialNeeded;
  elements.recoveryError.textContent = session?.pending_turn_error || "";
  elements.credentialPanel.hidden = !credentialNeeded || busy;
  elements.credentialTitle.textContent = credentialNeeded
    ? `配置 ${state.credentialRequest.providerID} API Key`
    : "配置 Provider API Key";
  elements.credentialError.textContent = credentialNeeded
    ? state.credentialRequest.error
    : "";
  elements.credentialPath.textContent = state.credentialsPath
    ? `将安全保存到 ${state.credentialsPath}`
    : "将安全保存到 Agent 凭据文件";
  elements.credentialKey.disabled = state.credentialSaving;
  elements.credentialSave.disabled = state.credentialSaving;
  elements.credentialAbort.disabled = state.credentialSaving;
}

function syncCredentialRequest(session) {
  const details = session?.pending_turn_error_details || {};
  if (
    session?.pending_turn_id &&
    details.action_required === "configure_api_key" &&
    details.provider_id
  ) {
    state.credentialRequest = {
      sessionID: session.session_id,
      providerID: details.provider_id,
      error: session.pending_turn_error || "API Key 无效或缺失",
    };
    return;
  }
  if (state.credentialRequest?.sessionID === session?.session_id) {
    state.credentialRequest = null;
  }
}

function conversationRoot() {
  let root = elements.conversation.querySelector(".conversation-inner");
  if (!root) {
    root = document.createElement("div");
    root.className = "conversation-inner";
    elements.conversation.replaceChildren(root);
  }
  return root;
}

function appendMessage(role, content, reasoning = "") {
  const article = document.createElement("article");
  article.className = `message message-${role}`;

  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = role === "user" ? "你" : "Agent";
  article.append(header);

  let reasoningBody = null;
  if (role === "assistant") {
    const details = document.createElement("details");
    details.className = "reasoning";
    details.hidden = !reasoning;
    const summary = document.createElement("summary");
    summary.textContent = "Thinking";
    reasoningBody = document.createElement("div");
    reasoningBody.className = "reasoning-body";
    reasoningBody.textContent = reasoning;
    details.append(summary, reasoningBody);
    article.append(details);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  if (role === "assistant") {
    renderMarkdown(body, content);
  } else {
    body.textContent = content;
  }
  article.append(body);
  conversationRoot().append(article);
  return { article, body, reasoningBody, reasoningDetails: reasoningBody?.parentElement || null };
}

function appendInline(parent, text) {
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else {
      const split = token.lastIndexOf("](");
      const link = document.createElement("a");
      link.textContent = token.slice(1, split);
      link.href = token.slice(split + 2, -1);
      link.target = "_blank";
      link.rel = "noreferrer noopener";
      parent.append(link);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function tableCells(line) {
  let value = line.trim();
  if (value.startsWith("|")) value = value.slice(1);
  if (value.endsWith("|")) value = value.slice(0, -1);
  return value.split("|").map((cell) => cell.trim());
}

function isTableDivider(line) {
  const cells = tableCells(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderMarkdown(container, markdown) {
  container.classList.add("rendered-markdown");
  const lines = String(markdown || "").replaceAll("\r\n", "\n").split("\n");
  const fragment = document.createDocumentFragment();
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.startsWith("```")) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = codeLines.join("\n");
      pre.append(code);
      fragment.append(pre);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const element = document.createElement(`h${heading[1].length}`);
      appendInline(element, heading[2]);
      fragment.append(element);
      index += 1;
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const headers = tableCells(line);
      const rows = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const value of headers) {
        const cell = document.createElement("th");
        appendInline(cell, value);
        headerRow.append(cell);
      }
      thead.append(headerRow);
      const tbody = document.createElement("tbody");
      for (const values of rows) {
        const row = document.createElement("tr");
        for (const value of values) {
          const cell = document.createElement("td");
          appendInline(cell, value);
          row.append(cell);
        }
        tbody.append(row);
      }
      table.append(thead, tbody);
      wrap.append(table);
      fragment.append(wrap);
      continue;
    }
    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.]\s+(.+)$/);
    if (unordered || ordered) {
      const list = document.createElement(unordered ? "ul" : "ol");
      const matcher = unordered ? /^\s*[-*]\s+(.+)$/ : /^\s*\d+[.]\s+(.+)$/;
      while (index < lines.length) {
        const item = lines[index].match(matcher);
        if (!item) break;
        const li = document.createElement("li");
        appendInline(li, item[1]);
        list.append(li);
        index += 1;
      }
      fragment.append(list);
      continue;
    }
    if (line.startsWith("> ")) {
      const quote = document.createElement("blockquote");
      const quoted = [];
      while (index < lines.length && lines[index].startsWith("> ")) {
        quoted.push(lines[index].slice(2));
        index += 1;
      }
      appendInline(quote, quoted.join("\n"));
      fragment.append(quote);
      continue;
    }
    if (/^---+$/.test(line.trim())) {
      fragment.append(document.createElement("hr"));
      index += 1;
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,4})\s+/.test(lines[index]) &&
      !lines[index].startsWith("```") &&
      !/^\s*[-*]\s+/.test(lines[index]) &&
      !/^\s*\d+[.]\s+/.test(lines[index]) &&
      !lines[index].startsWith("> ") &&
      !(lines[index].includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1]))
    ) {
      paragraphLines.push(lines[index]);
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInline(paragraph, paragraphLines.join("\n"));
    fragment.append(paragraph);
  }
  container.replaceChildren(fragment);
}

function renderMessages(messages) {
  state.currentAssistant = null;
  const root = document.createElement("div");
  root.className = "conversation-inner";
  elements.conversation.replaceChildren(root);

  const historicalActivities = [];
  for (const message of messages) {
    if (message.role === "tool") {
      historicalActivities.push({
        title: `${message.tool_name || "tool"} 已完成`,
        detail: message.content || "",
        tone: "success",
      });
      continue;
    }
    if (message.role === "assistant" && !message.content && !message.reasoning_content) {
      for (const call of message.tool_calls || []) {
        historicalActivities.push({
          title: `${call.name} 已请求`,
          detail: compactJSON(call.arguments),
          tone: "neutral",
        });
      }
      continue;
    }
    appendMessage(message.role, message.content || "", message.reasoning_content || "");
  }

  state.activities = historicalActivities.slice(-30);
  renderActivities();

  const live = state.liveTurns.get(selectedSessionID());
  if (live && state.busySessions.has(selectedSessionID())) {
    state.currentAssistant = appendMessage("assistant", live.displayContent, live.reasoning);
    state.currentAssistant.body.classList.add("stream-caret");
  }

  scrollConversation();
}

function scrollConversation() {
  window.requestAnimationFrame(() => {
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
  });
}

function compactJSON(value) {
  try {
    const text = JSON.stringify(value);
    return text.length > 240 ? `${text.slice(0, 237)}...` : text;
  } catch {
    return String(value);
  }
}

function addActivity(title, detail = "", tone = "neutral") {
  state.activities.push({ title, detail, tone });
  if (state.activities.length > 60) state.activities.splice(0, state.activities.length - 60);
  renderActivities();
}

function renderActivities() {
  const fragment = document.createDocumentFragment();
  for (const activity of [...state.activities].reverse()) {
    const item = document.createElement("div");
    item.className = "activity-item";
    item.dataset.tone = activity.tone;
    const dot = document.createElement("span");
    dot.className = "activity-dot";
    const content = document.createElement("div");
    const title = document.createElement("div");
    title.className = "activity-title";
    title.textContent = activity.title;
    content.append(title);
    if (activity.detail) {
      const detail = document.createElement("div");
      detail.className = "activity-detail";
      detail.textContent = activity.detail;
      detail.title = activity.detail;
      content.append(detail);
    }
    item.append(dot, content);
    fragment.append(item);
  }
  elements.activityList.replaceChildren(fragment);
}

function renderContext(context) {
  const total = Number(context.context_window_tokens || 0);
  const used = Number(context.estimated_tokens || 0);
  const percent = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  elements.contextPercent.textContent = `${Math.round(percent)}%`;
  elements.contextTokens.textContent = `${formatTokens(used)} / ${formatTokens(total)} tokens`;
  elements.usageFill.style.width = `${percent}%`;

  const breakdown = context.breakdown || {};
  const categories = [
    ["System prompt", breakdown.system_prompt_tokens, "#70787c"],
    ["Tool definitions", breakdown.tool_definition_tokens, "#286f9e"],
    ["Conversation summary", breakdown.summary_tokens, "#9a6500"],
    ["Conversation", breakdown.conversation_tokens, "#168153"],
    ["Protocol overhead", breakdown.protocol_tokens, "#a63c3c"],
  ];
  const fragment = document.createDocumentFragment();
  for (const [label, value, color] of categories) {
    const row = document.createElement("div");
    row.className = "context-row";
    const swatch = document.createElement("span");
    swatch.className = "context-swatch";
    swatch.style.setProperty("--swatch", color);
    const name = document.createElement("span");
    name.textContent = label;
    const count = document.createElement("span");
    count.className = "context-value";
    count.textContent = formatTokens(value);
    row.append(swatch, name, count);
    fragment.append(row);
  }
  elements.contextBreakdown.replaceChildren(fragment);
  elements.contextMeta.textContent = [
    `Input budget ${formatTokens(context.input_budget)}`,
    `Reserved output ${formatTokens(context.reserve_output_tokens)}`,
    `Active ${context.active_messages || 0} · Summarized ${context.summarized_messages || 0}`,
  ].join("\n");
}

function resetContext() {
  renderContext({ breakdown: {} });
}

async function refreshContext() {
  const sessionID = selectedSessionID();
  if (!sessionID) return resetContext();
  try {
    const result = await rpc("context.inspect", { session_id: sessionID });
    if (sessionID === selectedSessionID()) renderContext(result.context || {});
  } catch (error) {
    addActivity("上下文读取失败", error.message, "warning");
  }
}

async function refreshSessions() {
  const result = await rpc("session.list");
  state.sessions = result.sessions || [];
  renderSessionList();
}

async function selectSession(sessionID) {
  const summary = sessionSummary(sessionID);
  if (!summary) return;
  try {
    if (!summary.active) {
      await rpc("session.resume", { session_id: sessionID });
      summary.active = true;
    }
    const result = await rpc("session.read", { session_id: sessionID });
    state.selectedSession = result.session;
    syncCredentialRequest(state.selectedSession);
    state.attentionSessions.delete(sessionID);
    elements.approvalPanel.hidden = true;
    state.pendingApproval = null;
    renderMessages(result.messages || []);
    updateControls();
    renderSessionList();
    await refreshContext();
    closeOverlays();
  } catch (error) {
    showToast(`无法打开会话：${error.message}`);
  }
}

async function createSession() {
  const params = {};
  if (elements.modelSelect.value) params.model_ref = elements.modelSelect.value;
  const selectedMode = elements.modeSelector.querySelector('[aria-pressed="true"]');
  if (selectedMode?.dataset.modeId) params.mode_id = selectedMode.dataset.modeId;
  try {
    const result = await rpc("session.create", params);
    await refreshSessions();
    await selectSession(result.session.session_id);
    elements.promptInput.focus();
  } catch (error) {
    showToast(`无法新建会话：${error.message}`);
  }
}

function ensureLiveTurn(sessionID) {
  let live = state.liveTurns.get(sessionID);
  if (!live) {
    const session = sessionID === selectedSessionID() ? state.selectedSession : null;
    live = {
      rawContent: "",
      displayContent: "",
      reasoning: "",
      modeId: session?.mode_id || "build",
    };
    state.liveTurns.set(sessionID, live);
  }
  return live;
}

function visiblePlanContent(raw) {
  const startTag = "<proposed_plan>";
  const endTag = "</proposed_plan>";
  const start = raw.indexOf(startTag);
  if (start < 0) return "";
  let body = raw.slice(start + startTag.length);
  const end = body.indexOf(endTag);
  if (end >= 0) return body.slice(0, end);
  for (let length = Math.min(endTag.length - 1, body.length); length > 0; length -= 1) {
    if (endTag.startsWith(body.slice(-length))) return body.slice(0, -length);
  }
  return body;
}

function updateLiveAssistant(sessionID) {
  if (sessionID !== selectedSessionID()) return;
  const live = ensureLiveTurn(sessionID);
  if (!state.currentAssistant) {
    state.currentAssistant = appendMessage("assistant", "", "");
    state.currentAssistant.body.classList.add("stream-caret");
  }
  state.currentAssistant.body.classList.remove("rendered-markdown");
  state.currentAssistant.body.textContent = live.displayContent;
  state.currentAssistant.reasoningBody.textContent = live.reasoning;
  state.currentAssistant.reasoningDetails.hidden = !live.reasoning;
  scrollConversation();
}

async function startTurn(userInput) {
  const sessionID = selectedSessionID();
  if (!sessionID || !userInput.trim()) return;
  const text = userInput.trim();
  elements.promptInput.value = "";
  resizePrompt();
  appendMessage("user", text);
  state.liveTurns.set(sessionID, {
    rawContent: "",
    displayContent: "",
    reasoning: "",
    modeId: state.selectedSession.mode_id,
  });
  state.currentAssistant = null;
  state.credentialRequest = null;
  state.busySessions.add(sessionID);
  updateControls();
  renderSessionList();
  updateLiveAssistant(sessionID);
  try {
    await rpc("turn.start", { session_id: sessionID, user_input: text });
  } catch (error) {
    state.busySessions.delete(sessionID);
    updateControls();
    renderSessionList();
    addActivity("Turn 启动失败", error.message, "danger");
    showToast(error.message);
  }
}

async function switchMode(modeID) {
  const sessionID = selectedSessionID();
  if (!sessionID || modeID === state.selectedSession.mode_id) return;
  try {
    await rpc("mode.switch", { session_id: sessionID, mode_id: modeID });
  } catch (error) {
    showToast(`模式切换失败：${error.message}`);
  }
}

async function switchModel(modelRef) {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    await rpc("model.switch", { session_id: sessionID, model_ref: modelRef });
  } catch (error) {
    updateControls();
    showToast(`模型切换失败：${error.message}`);
  }
}

async function resolveApproval(decision) {
  const approval = state.pendingApproval;
  if (!approval || approval.sessionID !== selectedSessionID()) return;
  for (const button of elements.approvalPanel.querySelectorAll("button")) button.disabled = true;
  try {
    await rpc("approval.resolve", {
      session_id: approval.sessionID,
      approval_request_id: approval.requestID,
      decision,
    });
  } catch (error) {
    showToast(`审批失败：${error.message}`);
    for (const button of elements.approvalPanel.querySelectorAll("button")) button.disabled = false;
  }
}

function showApproval(sessionID, payload) {
  state.pendingApproval = { sessionID, requestID: payload.request_id };
  if (sessionID !== selectedSessionID()) {
    state.attentionSessions.add(sessionID);
    renderSessionList();
    return;
  }
  elements.approvalTitle.textContent = payload.description || `${payload.tool_name}: ${payload.action}`;
  const details = payload.details || {};
  if (details.patch_preview) {
    elements.approvalDetails.textContent = details.patch_preview;
  } else if (details.argv) {
    elements.approvalDetails.textContent = `${compactJSON(details.argv)}\ncwd: ${details.cwd || "."}`;
  } else {
    elements.approvalDetails.textContent = payload.resource || "";
  }
  for (const button of elements.approvalPanel.querySelectorAll("button")) button.disabled = false;
  elements.approvalPanel.hidden = false;
}

async function executePlan() {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    elements.planPanel.hidden = true;
    state.busySessions.add(sessionID);
    updateControls();
    renderSessionList();
    await rpc("plan.execute", { session_id: sessionID });
  } catch (error) {
    state.busySessions.delete(sessionID);
    updateControls();
    renderSessionList();
    showToast(`计划执行失败：${error.message}`);
  }
}

async function revisePlan() {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    await rpc("plan.revise", { session_id: sessionID });
    elements.planPanel.hidden = true;
    state.selectedSession.proposed_plan_status = "revising";
    updateControls();
    elements.promptInput.focus();
  } catch (error) {
    showToast(`计划修改失败：${error.message}`);
  }
}

async function retryTurn() {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    state.busySessions.add(sessionID);
    state.liveTurns.set(sessionID, {
      rawContent: "",
      displayContent: "",
      reasoning: "",
      modeId: state.selectedSession.mode_id,
    });
    updateControls();
    renderSessionList();
    await rpc("turn.retry", { session_id: sessionID });
  } catch (error) {
    state.busySessions.delete(sessionID);
    updateControls();
    renderSessionList();
    showToast(`重试失败：${error.message}`);
  }
}

async function saveCredential(event) {
  event.preventDefault();
  const request = state.credentialRequest;
  const apiKey = elements.credentialKey.value.trim();
  if (!request || request.sessionID !== selectedSessionID() || !apiKey) return;
  state.credentialSaving = true;
  updateControls();
  try {
    const result = await rpc("provider.credential.set", {
      session_id: request.sessionID,
      api_key: apiKey,
    });
    elements.credentialKey.value = "";
    state.credentialRequest = null;
    state.selectedSession = result.session;
    if (result.credentials_path) state.credentialsPath = result.credentials_path;
    addActivity(
      `${result.provider_id} 凭据已保存`,
      result.credentials_path || "Agent 凭据文件",
      "success",
    );
    showToast("API Key 已保存，正在重试当前 Turn");
  } catch (error) {
    showToast(`API Key 保存失败：${error.message}`);
    return;
  } finally {
    state.credentialSaving = false;
    updateControls();
  }
  await retryTurn();
}

async function abortTurn() {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    await rpc("turn.abort", { session_id: sessionID });
  } catch (error) {
    showToast(`放弃失败：${error.message}`);
  }
}

async function interruptTurn() {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    await rpc("turn.interrupt", { session_id: sessionID });
    addActivity("已请求停止", shortSessionID(sessionID), "warning");
  } catch (error) {
    showToast(`停止失败：${error.message}`);
  }
}

function eventLabel(sessionID, label) {
  return sessionID === selectedSessionID() ? label : `${shortSessionID(sessionID)} · ${label}`;
}

async function handleTerminalEvent(event, tone, detail) {
  const sessionID = event.session_id;
  const payload = event.payload || {};
  const live = ensureLiveTurn(sessionID);
  if (payload.answer) {
    live.rawContent = payload.answer;
    live.displayContent = payload.answer;
  }
  state.busySessions.delete(sessionID);
  if (sessionID === selectedSessionID()) {
    if (state.currentAssistant) {
      state.currentAssistant.body.classList.remove("stream-caret");
      state.currentAssistant.body.textContent = live.displayContent;
    }
    elements.approvalPanel.hidden = true;
    state.pendingApproval = null;
  }
  addActivity(eventLabel(sessionID, detail), "", tone);
  updateControls();
  renderSessionList();

  if (sessionID === selectedSessionID()) {
    try {
      const result = await rpc("session.read", { session_id: sessionID });
      state.selectedSession = result.session;
      syncCredentialRequest(state.selectedSession);
      state.liveTurns.delete(sessionID);
      renderMessages(result.messages || []);
      updateControls();
      await refreshContext();
    } catch (error) {
      addActivity("会话刷新失败", error.message, "warning");
    }
  }
  await refreshSessions().catch(() => {});
}

async function handleEvent(event) {
  if (event.channel !== "event" || event.protocol_version !== bootstrap.protocolVersion) return;
  const sessionID = event.session_id;
  const payload = event.payload || {};
  const previous = state.lastSequence.get(sessionID) || 0;
  if (event.sequence <= previous) return;
  if (previous && event.sequence > previous + 1) {
    addActivity(eventLabel(sessionID, "事件序号出现缺口"), `${previous} → ${event.sequence}`, "warning");
  }
  state.lastSequence.set(sessionID, event.sequence);

  switch (event.type) {
    case "turn.started": {
      const existing = ensureLiveTurn(sessionID);
      existing.modeId = sessionID === selectedSessionID() ? state.selectedSession.mode_id : existing.modeId;
      state.busySessions.add(sessionID);
      addActivity(eventLabel(sessionID, "Turn 已开始"), payload.turn_id || "", "neutral");
      break;
    }
    case "turn.resumed":
      state.busySessions.add(sessionID);
      addActivity(eventLabel(sessionID, "Turn 已恢复"), `step ${payload.step}`, "neutral");
      break;
    case "context.compacted":
      addActivity(eventLabel(sessionID, "上下文已压缩"), `step ${payload.step}`, "warning");
      break;
    case "model.started":
      addActivity(eventLabel(sessionID, "模型请求"), `step ${payload.step}`, "neutral");
      break;
    case "model.reasoning_delta": {
      const live = ensureLiveTurn(sessionID);
      live.reasoning += payload.text || "";
      updateLiveAssistant(sessionID);
      break;
    }
    case "model.content_delta": {
      const live = ensureLiveTurn(sessionID);
      live.rawContent += payload.text || "";
      live.displayContent = live.modeId === "plan" ? visiblePlanContent(live.rawContent) : live.rawContent;
      updateLiveAssistant(sessionID);
      break;
    }
    case "provider.retry":
      addActivity(
        eventLabel(sessionID, `Provider 重试 ${payload.attempt}`),
        `${payload.delay_seconds}s · ${payload.error}`,
        "warning",
      );
      break;
    case "tool.proposed":
      addActivity(eventLabel(sessionID, `${payload.name} 已提出`), compactJSON(payload.arguments), "neutral");
      break;
    case "tool.started":
      addActivity(eventLabel(sessionID, `${payload.name} 运行中`), compactJSON(payload.arguments), "neutral");
      break;
    case "tool.finished":
      addActivity(
        eventLabel(sessionID, `${payload.name} ${payload.ok ? "已完成" : "失败"}`),
        payload.duration_ms == null ? payload.status : `${payload.status} · ${payload.duration_ms}ms`,
        payload.ok ? "success" : "danger",
      );
      break;
    case "approval.requested":
      showApproval(sessionID, payload);
      addActivity(eventLabel(sessionID, "等待审批"), payload.description || payload.tool_name, "warning");
      break;
    case "approval.resolved":
      if (sessionID === selectedSessionID()) {
        elements.approvalPanel.hidden = true;
        state.pendingApproval = null;
      }
      state.attentionSessions.delete(sessionID);
      addActivity(eventLabel(sessionID, "审批已处理"), payload.decision, "neutral");
      break;
    case "turn.error":
      if (
        sessionID === selectedSessionID() &&
        payload.details?.action_required === "configure_api_key"
      ) {
        state.selectedSession.pending_turn_error = payload.error;
        state.selectedSession.pending_turn_retryable = payload.retryable;
        state.selectedSession.pending_turn_error_details = payload.details;
        syncCredentialRequest(state.selectedSession);
      }
      addActivity(eventLabel(sessionID, "模型错误"), payload.error, payload.retryable ? "warning" : "danger");
      break;
    case "turn.completed":
      if (payload.plan_id && sessionID === selectedSessionID()) {
        state.selectedSession.proposed_plan_id = payload.plan_id;
        state.selectedSession.proposed_plan_status = "proposed";
      }
      await handleTerminalEvent(event, "success", payload.plan_id ? "计划已生成" : "Turn 已完成");
      break;
    case "turn.failed":
      await handleTerminalEvent(event, "danger", `Turn 失败：${payload.error}`);
      break;
    case "turn.interrupted":
      await handleTerminalEvent(event, "warning", `Turn 已中断：${payload.reason}`);
      break;
    case "model.changed":
      if (sessionID === selectedSessionID()) {
        state.selectedSession.provider_id = payload.provider_id;
        state.selectedSession.model_id = payload.model_id;
        updateControls();
      }
      addActivity(eventLabel(sessionID, "模型已切换"), `${payload.provider_id}/${payload.model_id}`, "success");
      break;
    case "mode.changed":
      if (sessionID === selectedSessionID()) {
        state.selectedSession.mode_id = payload.mode_id;
        updateControls();
        refreshContext();
      }
      addActivity(eventLabel(sessionID, "模式已切换"), payload.mode_id, "success");
      break;
    case "plan.approved":
      if (sessionID === selectedSessionID()) {
        state.selectedSession.mode_id = payload.mode_id;
        state.selectedSession.proposed_plan_status = "approved";
        elements.planPanel.hidden = true;
      }
      addActivity(eventLabel(sessionID, "计划已批准"), payload.plan_id, "success");
      break;
    case "plan.revision_requested":
      addActivity(eventLabel(sessionID, "等待计划修改"), payload.plan_id, "warning");
      break;
    case "turn.aborted":
      state.busySessions.delete(sessionID);
      if (sessionID === selectedSessionID()) {
        state.selectedSession.pending_turn_id = null;
        state.liveTurns.delete(sessionID);
        state.credentialRequest = null;
        updateControls();
      }
      addActivity(eventLabel(sessionID, "Turn 已放弃"), payload.turn_id, "warning");
      await selectSession(sessionID);
      break;
    case "events.replay_completed":
      addActivity(eventLabel(sessionID, "事件重放完成"), `${payload.replayed_events} events`, "neutral");
      break;
    default:
      addActivity(eventLabel(sessionID, event.type), "", "neutral");
  }
  renderSessionList();
  updateControls();
}

async function pollEvents() {
  let failures = 0;
  while (!state.stopped) {
    try {
      const response = await fetch("/api/events?timeout=20", {
        headers: { "X-My-Agent-Token": bootstrap.token },
        cache: "no-store",
      });
      if (response.status === 204) {
        setConnection("已连接", "online");
        failures = 0;
        continue;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setConnection("已连接", "online");
      failures = 0;
      await handleEvent(await response.json());
    } catch (error) {
      failures += 1;
      setConnection("连接中断", "offline");
      if (failures === 1) showToast(`事件连接中断：${error.message}`);
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(5000, failures * 500)));
    }
  }
}

function resizePrompt() {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(elements.promptInput.scrollHeight, 180)}px`;
}

function closeOverlays() {
  elements.body.classList.remove("sidebar-open", "inspector-open");
  elements.backdrop.hidden = true;
}

function openOverlay(name) {
  closeOverlays();
  elements.body.classList.add(name);
  elements.backdrop.hidden = false;
}

function bindUI() {
  elements.workspacePath.textContent = bootstrap.workspace;
  elements.workspacePath.title = bootstrap.workspace;
  elements.newSession.addEventListener("click", createSession);
  elements.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    startTurn(elements.promptInput.value);
  });
  elements.promptInput.addEventListener("input", resizePrompt);
  elements.promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.composer.requestSubmit();
    }
  });
  elements.modelSelect.addEventListener("change", () => switchModel(elements.modelSelect.value));
  elements.stopTurn.addEventListener("click", interruptTurn);
  elements.executePlan.addEventListener("click", executePlan);
  elements.revisePlan.addEventListener("click", revisePlan);
  elements.retryTurn.addEventListener("click", retryTurn);
  elements.abortTurn.addEventListener("click", abortTurn);
  elements.credentialPanel.addEventListener("submit", saveCredential);
  elements.credentialAbort.addEventListener("click", abortTurn);
  elements.clearActivity.addEventListener("click", () => {
    state.activities = [];
    renderActivities();
  });
  for (const button of elements.approvalPanel.querySelectorAll("[data-approval]")) {
    button.addEventListener("click", () => resolveApproval(button.dataset.approval));
  }
  elements.sidebarToggle.addEventListener("click", () => openOverlay("sidebar-open"));
  elements.inspectorToggle.addEventListener("click", () => openOverlay("inspector-open"));
  elements.inspectorClose.addEventListener("click", closeOverlays);
  elements.backdrop.addEventListener("click", closeOverlays);
  window.addEventListener("beforeunload", () => {
    state.stopped = true;
  });
}

async function initialize() {
  if (!bootstrap?.token) {
    setConnection("初始化失败", "offline");
    showToast("缺少本地 UI bootstrap 配置");
    return;
  }
  bindUI();
  resetContext();
  pollEvents();
  try {
    const [models, modes, sessions] = await Promise.all([
      rpc("model.list"),
      rpc("mode.list"),
      rpc("session.list"),
    ]);
    state.models = models.models || [];
    state.credentialsPath = models.credentials_path || "";
    state.modes = modes.modes || [];
    state.sessions = sessions.sessions || [];
    renderModelCatalog();
    renderModeCatalog();
    renderSessionList();
    setConnection("已连接", "online");
    if (state.sessions.length) await selectSession(state.sessions[0].session_id);
  } catch (error) {
    setConnection("初始化失败", "offline");
    showToast(`初始化失败：${error.message}`);
  }
}

initialize();
