"use strict";

const bootstrap = window.__MY_AGENT__;

const elements = {
  body: document.body,
  workspacePath: document.querySelector("#workspace-path"),
  sessionList: document.querySelector("#session-list"),
  sessionTitle: document.querySelector("#session-title"),
  sessionTitleText: document.querySelector("#session-title-text"),
  sessionRenameForm: document.querySelector("#session-rename-form"),
  sessionTitleInput: document.querySelector("#session-title-input"),
  sessionRenameCancel: document.querySelector("#session-rename-cancel"),
  sessionRenameSave: document.querySelector("#session-rename-save"),
  sessionContextMenu: document.querySelector("#session-context-menu"),
  sessionMenuRename: document.querySelector("#session-menu-rename"),
  connectionStatus: document.querySelector("#connection-status"),
  conversation: document.querySelector("#conversation"),
  emptyState: document.querySelector("#empty-state"),
  newSession: document.querySelector("#new-session"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  inspectorToggle: document.querySelector("#inspector-toggle"),
  inspectorClose: document.querySelector("#inspector-close"),
  backdrop: document.querySelector("#backdrop"),
  modeSelector: document.querySelector("#mode-selector"),
  sandboxMenuTrigger: document.querySelector("#sandbox-menu-trigger"),
  sandboxMenuValue: document.querySelector("#sandbox-menu-value"),
  sandboxMenu: document.querySelector("#sandbox-menu"),
  modelMenuTrigger: document.querySelector("#model-menu-trigger"),
  modelMenuValue: document.querySelector("#model-menu-value"),
  modelMenu: document.querySelector("#model-menu"),
  permissionMenuTrigger: document.querySelector("#permission-menu-trigger"),
  permissionMenuValue: document.querySelector("#permission-menu-value"),
  permissionMenu: document.querySelector("#permission-menu"),
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
  recoveryLabel: document.querySelector("#recovery-label"),
  recoveryTitle: document.querySelector("#recovery-title"),
  recoveryError: document.querySelector("#recovery-error"),
  recoveryAttempts: document.querySelector("#recovery-attempts"),
  recoveryConfirmation: document.querySelector("#recovery-confirmation"),
  recoveryConfirmed: document.querySelector("#recovery-confirmed"),
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
  goalView: document.querySelector("#goal-view"),
  clearGoal: document.querySelector("#clear-goal"),
  permissionGrantList: document.querySelector("#permission-grant-list"),
  clearPermissionGrants: document.querySelector("#clear-permission-grants"),
  activityList: document.querySelector("#activity-list"),
  clearActivity: document.querySelector("#clear-activity"),
  toast: document.querySelector("#toast"),
};

const state = {
  sessions: [],
  models: [],
  modes: [],
  permissionProfiles: [],
  sandboxProfiles: [],
  currentGoal: null,
  permissionGrants: [],
  selectedSession: null,
  busySessions: new Set(),
  attentionSessions: new Set(),
  lastSequence: new Map(),
  liveItems: new Map(),
  itemElements: new Map(),
  activities: [],
  subscriptionID: "",
  pendingApproval: null,
  credentialRequest: null,
  credentialsPath: "",
  credentialSaving: false,
  sessionMenuSessionID: null,
  renamingSessionID: null,
  reloadingServer: false,
  stopped: false,
};

let toastTimer = null;

const permissionLabels = {
  supervised: "监督",
  accept_edits: "自动编辑",
  trusted: "可信",
};

const sandboxLabels = {
  "workspace-write": "工作区写入",
  "danger-full-access": "完全访问",
};

function requestID() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `request-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function verifyServerInstance(response) {
  const responseInstanceID = response.headers.get("X-My-Agent-Instance");
  if (
    !responseInstanceID ||
    !bootstrap.instanceID ||
    responseInstanceID === bootstrap.instanceID
  ) {
    return;
  }
  if (!state.reloadingServer) {
    state.reloadingServer = true;
    state.stopped = true;
    setConnection("服务已重启", "offline");
    showToast("Agent 服务已重启，正在重新连接");
    window.setTimeout(() => window.location.reload(), 300);
  }
  throw new Error("Agent server instance changed");
}

async function rpc(method, params = {}) {
  const request = {
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
  };
  let response;
  try {
    response = await fetch("/api/rpc", request);
  } catch (error) {
    if (!bootstrap.deduplicatedMethods?.includes(method)) throw error;
    // A transport retry must retain the same request ID and body.
    response = await fetch("/api/rpc", request);
  }
  verifyServerInstance(response);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const envelope = await response.json();
  if (!envelope.ok) {
    const error = new Error(envelope.error?.message || "RPC failed");
    if (envelope.error?.code === "request_outcome_unknown") {
      error.message = "请求已登记，但执行结果尚不明确。请检查会话状态，不要重复提交新任务。";
    }
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

function sessionDisplayName(session) {
  return session?.title?.trim() || shortSessionID(session?.session_id || "");
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
    const row = document.createElement("div");
    row.className = "session-row";
    row.dataset.sessionId = session.session_id;
    if (session.session_id === selectedSessionID()) row.dataset.current = "true";

    if (state.renamingSessionID === session.session_id) {
      const form = document.createElement("form");
      form.className = "session-inline-rename";
      const input = document.createElement("input");
      input.type = "text";
      input.maxLength = 80;
      input.value = session.title || sessionDisplayName(session);
      input.setAttribute("aria-label", "会话名称");
      input.autocomplete = "off";
      input.spellcheck = false;
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.textContent = "×";
      cancel.title = "取消重命名";
      cancel.setAttribute("aria-label", "取消重命名");
      const save = document.createElement("button");
      save.type = "submit";
      save.textContent = "✓";
      save.title = "保存名称";
      save.setAttribute("aria-label", "保存名称");
      form.append(input, cancel, save);
      form.addEventListener("submit", (event) =>
        renameSidebarSession(event, session.session_id, input, save),
      );
      cancel.addEventListener("click", cancelSidebarRename);
      input.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          cancelSidebarRename();
        }
      });
      row.append(form);
      fragment.append(row);
      continue;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.dataset.sessionId = session.session_id;
    if (session.session_id === selectedSessionID()) {
      button.setAttribute("aria-current", "page");
    }

    const name = document.createElement("span");
    name.className = "session-name";
    name.textContent = sessionDisplayName(session);
    name.title = session.title || session.session_id;

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
    button.addEventListener("click", () => {
      closeSessionMenu();
      selectSession(session.session_id);
    });
    button.addEventListener("keydown", (event) => {
      if (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10")) {
        event.preventDefault();
        openSessionMenu(session.session_id, { anchor: more, focus: true });
      }
    });

    const more = document.createElement("button");
    more.type = "button";
    more.className = "session-actions";
    more.textContent = "⋯";
    more.title = `“${sessionDisplayName(session)}”的操作`;
    more.setAttribute("aria-label", `“${sessionDisplayName(session)}”的操作`);
    more.setAttribute("aria-haspopup", "menu");
    more.setAttribute("aria-expanded", "false");
    more.setAttribute("aria-controls", "session-context-menu");
    more.addEventListener("click", (event) => {
      event.stopPropagation();
      if (
        !elements.sessionContextMenu.hidden &&
        state.sessionMenuSessionID === session.session_id
      ) {
        closeSessionMenu();
        return;
      }
      openSessionMenu(session.session_id, { anchor: more, focus: false });
    });

    row.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      openSessionMenu(session.session_id, {
        point: { x: event.clientX, y: event.clientY },
        focus: true,
      });
    });
    row.append(button, more);
    fragment.append(row);
  }
  elements.sessionList.replaceChildren(fragment);
}

function openSessionMenu(sessionID, { anchor = null, point = null, focus = false } = {}) {
  closeChoiceMenus();
  closeSessionMenu();
  state.sessionMenuSessionID = sessionID;
  elements.sessionContextMenu.hidden = false;
  const menuRect = elements.sessionContextMenu.getBoundingClientRect();
  const anchorRect = anchor?.getBoundingClientRect();
  const desiredLeft = point ? point.x : anchorRect.right - menuRect.width;
  const desiredTop = point ? point.y : anchorRect.bottom + 4;
  const left = Math.max(8, Math.min(desiredLeft, window.innerWidth - menuRect.width - 8));
  let top = Math.max(8, Math.min(desiredTop, window.innerHeight - menuRect.height - 8));
  if (!point && top < desiredTop) top = Math.max(8, anchorRect.top - menuRect.height - 4);
  elements.sessionContextMenu.style.left = `${left}px`;
  elements.sessionContextMenu.style.top = `${top}px`;
  for (const button of elements.sessionList.querySelectorAll(".session-actions")) {
    button.setAttribute(
      "aria-expanded",
      String(button.closest(".session-row")?.dataset.sessionId === sessionID),
    );
  }
  if (focus) elements.sessionMenuRename.focus();
}

function closeSessionMenu() {
  if (!elements.sessionContextMenu.hidden) elements.sessionContextMenu.hidden = true;
  state.sessionMenuSessionID = null;
  for (const button of elements.sessionList.querySelectorAll(".session-actions")) {
    button.setAttribute("aria-expanded", "false");
  }
}

function renderModelCatalog() {
  const fragment = document.createDocumentFragment();
  for (const model of state.models) {
    fragment.append(
      createChoiceOption({
        value: model.ref,
        label: model.ref,
        description: model.description,
        onSelect: switchModel,
      }),
    );
  }
  elements.modelMenu.replaceChildren(fragment);
  if (!elements.modelMenuTrigger.dataset.value && state.models[0]) {
    setChoiceMenuValue(
      elements.modelMenuTrigger,
      elements.modelMenuValue,
      elements.modelMenu,
      state.models[0].ref,
      state.models[0].model_id || state.models[0].ref,
    );
  }
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

function renderPermissionCatalog() {
  const fragment = document.createDocumentFragment();
  for (const profile of state.permissionProfiles) {
    fragment.append(
      createChoiceOption({
        value: profile.profile_id,
        label: permissionLabels[profile.profile_id] || profile.profile_id,
        description: profile.description,
        onSelect: switchPermissionProfile,
      }),
    );
  }
  elements.permissionMenu.replaceChildren(fragment);
  if (!elements.permissionMenuTrigger.dataset.value && state.permissionProfiles[0]) {
    const profile = state.permissionProfiles[0];
    setChoiceMenuValue(
      elements.permissionMenuTrigger,
      elements.permissionMenuValue,
      elements.permissionMenu,
      profile.profile_id,
      permissionLabels[profile.profile_id] || profile.profile_id,
    );
  }
}

function renderSandboxCatalog() {
  const fragment = document.createDocumentFragment();
  for (const profile of state.sandboxProfiles) {
    const availability = profile.available
      ? profile.description
      : `${profile.description} 不可用：${profile.unavailable_reason || "缺少 backend"}`;
    fragment.append(
      createChoiceOption({
        value: profile.mode,
        label: sandboxLabels[profile.mode] || profile.mode,
        description: profile.dangerous ? `高风险：${availability}` : availability,
        disabled: !profile.available,
        onSelect: switchSandbox,
      }),
    );
  }
  elements.sandboxMenu.replaceChildren(fragment);
  if (!elements.sandboxMenuTrigger.dataset.value) {
    const profile = state.sandboxProfiles[0];
    if (profile) {
      setChoiceMenuValue(
        elements.sandboxMenuTrigger,
        elements.sandboxMenuValue,
        elements.sandboxMenu,
        profile.mode,
        sandboxLabels[profile.mode] || profile.mode,
      );
    }
  }
}

function createChoiceOption({ value, label, description, disabled = false, onSelect }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "choice-menu-option";
  button.dataset.value = value;
  button.setAttribute("role", "option");
  button.setAttribute("aria-selected", "false");
  button.disabled = disabled;

  const copy = document.createElement("span");
  copy.className = "choice-menu-option-copy";
  const primary = document.createElement("strong");
  primary.textContent = label;
  copy.append(primary);
  if (description) {
    const secondary = document.createElement("span");
    secondary.textContent = description;
    copy.append(secondary);
  }
  const check = document.createElement("span");
  check.className = "choice-menu-check";
  check.textContent = "✓";
  check.setAttribute("aria-hidden", "true");
  button.append(copy, check);
  button.addEventListener("click", () => {
    closeChoiceMenus();
    onSelect(value);
  });
  return button;
}

function setChoiceMenuValue(trigger, valueElement, menu, value, label) {
  trigger.dataset.value = value || "";
  valueElement.textContent = label || "未选择";
  valueElement.title = value || "";
  for (const option of menu.querySelectorAll(".choice-menu-option")) {
    const selected = option.dataset.value === value;
    option.setAttribute("aria-selected", String(selected));
  }
}

function closeChoiceMenus(except = null) {
  for (const [trigger, menu] of [
    [elements.sandboxMenuTrigger, elements.sandboxMenu],
    [elements.modelMenuTrigger, elements.modelMenu],
    [elements.permissionMenuTrigger, elements.permissionMenu],
  ]) {
    if (menu === except) continue;
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
  }
}

function openChoiceMenu(trigger, menu, direction = 1) {
  if (trigger.disabled) return;
  closeChoiceMenus(menu);
  menu.hidden = false;
  trigger.setAttribute("aria-expanded", "true");
  const options = [...menu.querySelectorAll(".choice-menu-option:not(:disabled)")];
  const selected = options.find((option) => option.getAttribute("aria-selected") === "true");
  const target = selected || (direction < 0 ? options.at(-1) : options[0]);
  target?.focus();
}

function bindChoiceMenu(trigger, menu) {
  trigger.addEventListener("click", () => {
    if (menu.hidden) openChoiceMenu(trigger, menu);
    else closeChoiceMenus();
  });
  trigger.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      openChoiceMenu(trigger, menu, event.key === "ArrowUp" ? -1 : 1);
    }
  });
  menu.addEventListener("keydown", (event) => {
    const options = [...menu.querySelectorAll(".choice-menu-option:not(:disabled)")];
    const index = options.indexOf(document.activeElement);
    if (event.key === "Escape") {
      event.preventDefault();
      closeChoiceMenus();
      trigger.focus();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const offset = event.key === "ArrowDown" ? 1 : -1;
      options[(index + offset + options.length) % options.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      options[event.key === "Home" ? 0 : options.length - 1]?.focus();
    }
  });
}

function updateControls() {
  const session = state.selectedSession;
  const sessionID = selectedSessionID();
  const busy = sessionID ? state.busySessions.has(sessionID) : false;
  const pending = Boolean(session?.pending_turn_id);
  const storageFailure = session?.persistence_error;
  const credentialNeeded = Boolean(
    pending && state.credentialRequest?.sessionID === sessionID,
  );
  const enabled = Boolean(session) && !storageFailure;

  elements.sessionTitleText.textContent = session ? sessionDisplayName(session) : "未选择会话";
  elements.sessionTitle.title = session ? `重命名“${sessionDisplayName(session)}”` : "重命名会话";
  elements.sessionTitle.disabled = !enabled;
  elements.promptInput.disabled = !enabled || busy || pending;
  elements.sendPrompt.disabled = !enabled || busy || pending;
  elements.stopTurn.hidden = !busy;
  elements.modelMenuTrigger.disabled = !enabled || busy || pending;
  elements.permissionMenuTrigger.disabled = !enabled || busy || pending;
  elements.sandboxMenuTrigger.disabled = busy || pending || !state.sandboxProfiles.length;

  if (session?.provider_id && session?.model_id) {
    const modelRef = `${session.provider_id}/${session.model_id}`;
    const model = state.models.find((item) => item.ref === modelRef);
    setChoiceMenuValue(
      elements.modelMenuTrigger,
      elements.modelMenuValue,
      elements.modelMenu,
      modelRef,
      model?.model_id || session.model_id,
    );
  }
  if (session?.permission_profile_id) {
    setChoiceMenuValue(
      elements.permissionMenuTrigger,
      elements.permissionMenuValue,
      elements.permissionMenu,
      session.permission_profile_id,
      permissionLabels[session.permission_profile_id] || session.permission_profile_id,
    );
  }
  if (session?.sandbox_mode) {
    setChoiceMenuValue(
      elements.sandboxMenuTrigger,
      elements.sandboxMenuValue,
      elements.sandboxMenu,
      session.sandbox_mode,
      sandboxLabels[session.sandbox_mode] || session.sandbox_mode,
    );
  }
  for (const button of elements.modeSelector.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.modeId === session?.mode_id));
    button.disabled = !enabled || busy || pending;
  }

  const proposed = session?.proposed_plan_status === "proposed";
  const continuationRequired = session?.pending_turn_pause_reason === "step_budget";
  const recoveryRequired = session?.pending_turn_pause_reason === "recovery_required";
  const pauseDetails = session?.pending_turn_pause_details || {};
  elements.planPanel.hidden = !proposed || busy;
  elements.recoveryPanel.hidden = (!pending && !storageFailure) || busy || credentialNeeded;
  elements.recoveryLabel.textContent = recoveryRequired ? "上次运行意外中断" : continuationRequired ? "需要你的决定" : "Turn 未完成";
  elements.recoveryTitle.textContent = recoveryRequired ? "检查工作区后恢复"
    : continuationRequired
    ? `已完成 ${pauseDetails.steps_completed || 0} 个模型步骤`
    : "重试或放弃当前 turn";
  const unknownAttempts = pauseDetails.unknown_attempts || [];
  elements.recoveryError.textContent = recoveryRequired
    ? unknownAttempts.length
      ? `${unknownAttempts.length} 个工具的执行结果未知，工作区可能已经被修改。原调用不会自动重放。`
      : "上次运行未正常结束。已完成的修改不会撤销，请确认工作区现状。"
    : continuationRequired
    ? "模型仍希望继续调用工具。当前状态已保存，是否继续执行？"
    : session?.pending_turn_error || "";
  const confirmationKey = `${sessionID}:${session?.pending_turn_id}`;
  if (!recoveryRequired || elements.recoveryConfirmed.dataset.turn !== confirmationKey) {
    elements.recoveryConfirmed.checked = false;
    elements.recoveryConfirmed.dataset.turn = confirmationKey;
  }
  elements.recoveryConfirmation.hidden = !recoveryRequired;
  elements.recoveryAttempts.hidden = !recoveryRequired || !unknownAttempts.length;
  elements.recoveryAttempts.replaceChildren(...unknownAttempts.map((attempt) => {
    const item = document.createElement("li");
    const details = document.createElement("details");
    const label = document.createElement("summary");
    label.textContent = `${attempt.tool_name} · ${attempt.call_id}`;
    const argumentsView = document.createElement("pre");
    argumentsView.textContent = JSON.stringify(attempt.arguments, null, 2);
    details.append(label, argumentsView);
    item.append(details);
    return item;
  }));
  elements.retryTurn.disabled = busy || (recoveryRequired && !elements.recoveryConfirmed.checked);
  elements.retryTurn.textContent = recoveryRequired ? "确认恢复" : continuationRequired
    ? `继续 ${pauseDetails.step_budget || 0} 步`
    : "重试";
  elements.retryTurn.hidden = Boolean(storageFailure);
  elements.abortTurn.hidden = Boolean(storageFailure);
  if (storageFailure) {
    elements.recoveryLabel.textContent = "存储故障";
    elements.recoveryTitle.textContent = "修复存储后重启服务";
    elements.recoveryError.textContent = storageFailure;
    elements.recoveryConfirmation.hidden = true;
  }
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
  renderGoal();
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

function appendMessage(role, content, reasoning = "", itemKey = "") {
  const article = document.createElement("article");
  article.className = `message message-${role}`;
  if (itemKey) article.dataset.itemKey = itemKey;

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
  state.itemElements.clear();
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

  renderLiveItems(selectedSessionID());

  scrollConversation();
}

function loadProjection(projection) {
  if (!projection?.session_id) return;
  const sessionID = projection.session_id;
  state.lastSequence.set(sessionID, Number(projection.latest_sequence || 0));
  const turns = projection.turns || [];
  const activeTurn = [...turns]
    .reverse()
    .find((turn) => !["completed", "aborted"].includes(turn.status));
  const items = new Map();
  for (const item of activeTurn?.items || []) items.set(item.item_id, { ...item });
  state.liveItems.set(sessionID, items);
  const runtimeBusy = ["running_model", "waiting_approval", "running_tool"].includes(
    state.selectedSession?.runtime_state,
  );
  if (runtimeBusy) state.busySessions.add(sessionID);
  else state.busySessions.delete(sessionID);
}

function liveItemsFor(sessionID) {
  let items = state.liveItems.get(sessionID);
  if (!items) {
    items = new Map();
    state.liveItems.set(sessionID, items);
  }
  return items;
}

function renderLiveItems(sessionID) {
  if (!sessionID || sessionID !== selectedSessionID()) return;
  const groups = new Map();
  for (const item of liveItemsFor(sessionID).values()) {
    if (!["assistant_message", "reasoning", "plan"].includes(item.item_type)) continue;
    const key = `${item.turn_id}:step:${item.step}`;
    const group = groups.get(key) || { key, step: item.step };
    if (item.item_type === "reasoning") group.reasoning = item;
    else if (item.item_type === "plan") group.plan = item;
    else group.assistant = item;
    groups.set(key, group);
  }

  for (const group of [...groups.values()].sort((left, right) => left.step - right.step)) {
    let view = state.itemElements.get(group.key);
    if (!view) {
      view = appendMessage("assistant", "", "", group.key);
      state.itemElements.set(group.key, view);
    }
    const answerItem = group.plan || group.assistant;
    let content = answerItem?.content || "";
    if (!group.plan && state.selectedSession?.mode_id === "plan") {
      content = visiblePlanContent(content);
    }
    const streaming = [group.assistant, group.reasoning, group.plan].some(
      (item) => item?.status === "in_progress",
    );
    if (streaming) {
      view.body.classList.remove("rendered-markdown");
      view.body.textContent = content;
      view.body.classList.add("stream-caret");
    } else {
      view.body.classList.remove("stream-caret");
      renderMarkdown(view.body, content);
    }
    const reasoning = group.reasoning?.content || "";
    view.reasoningBody.textContent = reasoning;
    view.reasoningDetails.hidden = !reasoning;
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

function renderGoal() {
  const fragment = document.createDocumentFragment();
  const sessionID = selectedSessionID();
  const busy = sessionID ? state.busySessions.has(sessionID) : false;
  const pending = Boolean(state.selectedSession?.pending_turn_id);
  elements.clearGoal.disabled = !state.currentGoal || busy || pending;
  if (!state.selectedSession || !state.currentGoal) {
    const empty = document.createElement("div");
    empty.className = "goal-empty";
    empty.textContent = state.selectedSession ? "没有当前目标" : "选择会话后显示";
    fragment.append(empty);
  } else {
    const objective = document.createElement("strong");
    objective.className = "goal-objective";
    objective.textContent = state.currentGoal.objective;
    const status = document.createElement("span");
    status.className = "goal-status";
    status.dataset.status = state.currentGoal.status;
    status.textContent = state.currentGoal.status === "completed" ? "已完成" : "进行中";
    const steps = document.createElement("div");
    steps.className = "goal-steps";
    for (const goalStep of state.currentGoal.steps || []) {
      const step = document.createElement("div");
      step.className = "goal-step";
      step.dataset.status = goalStep.status;
      step.textContent = goalStep.description;
      steps.append(step);
    }
    fragment.append(objective, status, steps);
  }
  elements.goalView.replaceChildren(fragment);
}

function renderPermissionGrants() {
  const fragment = document.createDocumentFragment();
  elements.clearPermissionGrants.disabled = !state.selectedSession || !state.permissionGrants.length;
  if (!state.selectedSession || !state.permissionGrants.length) {
    const empty = document.createElement("div");
    empty.className = "permission-grant-empty";
    empty.textContent = state.selectedSession ? "没有本次运行授权" : "选择会话后显示";
    fragment.append(empty);
  } else {
    for (const grant of state.permissionGrants) {
      const row = document.createElement("div");
      row.className = "permission-grant-row";
      const copy = document.createElement("div");
      copy.className = "permission-grant-copy";
      const action = document.createElement("strong");
      action.textContent = grant.action;
      const resource = document.createElement("span");
      resource.textContent = grant.resource;
      resource.title = grant.description || grant.resource;
      copy.append(action, resource);
      const revoke = document.createElement("button");
      revoke.className = "icon-button permission-grant-revoke";
      revoke.type = "button";
      revoke.textContent = "×";
      revoke.title = `撤销 ${grant.action} ${grant.resource}`;
      revoke.setAttribute("aria-label", revoke.title);
      revoke.addEventListener("click", () => revokePermissionGrant(grant));
      row.append(copy, revoke);
      fragment.append(row);
    }
  }
  elements.permissionGrantList.replaceChildren(fragment);
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

async function refreshPermissionGrants() {
  const sessionID = selectedSessionID();
  if (!sessionID) {
    state.permissionGrants = [];
    renderPermissionGrants();
    return;
  }
  try {
    const result = await rpc("permission.grants.list", { session_id: sessionID });
    if (sessionID === selectedSessionID()) {
      state.permissionGrants = result.grants || [];
      renderPermissionGrants();
    }
  } catch (error) {
    addActivity("运行时授权读取失败", error.message, "warning");
  }
}

async function refreshGoal() {
  const sessionID = selectedSessionID();
  if (!sessionID) {
    state.currentGoal = null;
    renderGoal();
    return;
  }
  try {
    const result = await rpc("goal.read", { session_id: sessionID });
    if (sessionID === selectedSessionID()) {
      state.currentGoal = result.goal || null;
      renderGoal();
    }
  } catch (error) {
    addActivity("目标读取失败", error.message, "warning");
  }
}

async function clearGoal() {
  const sessionID = selectedSessionID();
  if (!sessionID || !state.currentGoal) return;
  try {
    await rpc("goal.clear", { session_id: sessionID });
    await refreshGoal();
  } catch (error) {
    showToast(`无法清除目标：${error.message}`);
  }
}

async function revokePermissionGrant(grant) {
  const sessionID = selectedSessionID();
  if (!sessionID) return;
  try {
    await rpc("permission.grant.revoke", {
      session_id: sessionID,
      action: grant.action,
      resource: grant.resource,
    });
    await refreshPermissionGrants();
  } catch (error) {
    showToast(`无法撤销授权：${error.message}`);
  }
}

async function clearPermissionGrants() {
  const sessionID = selectedSessionID();
  if (!sessionID || !state.permissionGrants.length) return;
  try {
    await rpc("permission.grants.clear", { session_id: sessionID });
    await refreshPermissionGrants();
  } catch (error) {
    showToast(`无法清空授权：${error.message}`);
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
  state.renamingSessionID = null;
  closeSessionMenu();
  cancelSessionRename();
  try {
    if (!summary.active) {
      await rpc("session.resume", { session_id: sessionID });
      summary.active = true;
    }
    const result = await rpc("session.read", { session_id: sessionID });
    state.selectedSession = result.session;
    state.currentGoal = null;
    renderGoal();
    state.permissionGrants = [];
    renderPermissionGrants();
    loadProjection(result.projection);
    syncCredentialRequest(state.selectedSession);
    state.attentionSessions.delete(sessionID);
    elements.approvalPanel.hidden = true;
    state.pendingApproval = null;
    renderMessages(result.messages || []);
    updateControls();
    renderSessionList();
    await refreshContext();
    await refreshGoal();
    await refreshPermissionGrants();
    closeOverlays();
  } catch (error) {
    showToast(`无法打开会话：${error.message}`);
  }
}

async function createSession() {
  const params = {};
  if (elements.modelMenuTrigger.dataset.value) {
    params.model_ref = elements.modelMenuTrigger.dataset.value;
  }
  if (elements.permissionMenuTrigger.dataset.value) {
    params.permission_profile_id = elements.permissionMenuTrigger.dataset.value;
  }
  if (elements.sandboxMenuTrigger.dataset.value) {
    params.sandbox_mode = elements.sandboxMenuTrigger.dataset.value;
  }
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

function beginSessionRename() {
  const session = state.selectedSession;
  if (!session) return;
  closeChoiceMenus();
  elements.sessionTitleInput.value = session.title || sessionDisplayName(session);
  elements.sessionTitle.hidden = true;
  elements.sessionRenameForm.hidden = false;
  elements.sessionTitleInput.focus();
  elements.sessionTitleInput.select();
}

async function beginSidebarRename(sessionID) {
  closeSessionMenu();
  const summary = sessionSummary(sessionID);
  if (!summary) return;
  try {
    if (!summary.active) {
      await rpc("session.resume", { session_id: sessionID });
      summary.active = true;
    }
    state.renamingSessionID = sessionID;
    renderSessionList();
    const input = elements.sessionList.querySelector(".session-inline-rename input");
    input?.focus();
    input?.select();
  } catch (error) {
    showToast(`无法重命名会话：${error.message}`);
  }
}

function cancelSidebarRename() {
  const sessionID = state.renamingSessionID;
  state.renamingSessionID = null;
  renderSessionList();
  elements.sessionList
    .querySelector(`.session-row[data-session-id="${sessionID}"] .session-actions`)
    ?.focus();
}

function validateSessionTitle(title) {
  if (!title) return "会话名称不能为空";
  if (title.length > 80) return "会话名称最多 80 个字符";
  return "";
}

async function saveSessionTitle(sessionID, title) {
  await rpc("session.rename", { session_id: sessionID, title });
  updateSessionSummary(sessionID, { title });
  if (sessionID === selectedSessionID()) {
    state.selectedSession.title = title;
    updateControls();
  }
}

async function renameSidebarSession(event, sessionID, input, saveButton) {
  event.preventDefault();
  const title = input.value.trim();
  const error = validateSessionTitle(title);
  if (error) {
    showToast(error);
    input.focus();
    return;
  }
  if (title === sessionSummary(sessionID)?.title) {
    cancelSidebarRename();
    return;
  }
  input.disabled = true;
  saveButton.disabled = true;
  try {
    await saveSessionTitle(sessionID, title);
    state.renamingSessionID = null;
    renderSessionList();
  } catch (rpcError) {
    input.disabled = false;
    saveButton.disabled = false;
    input.focus();
    showToast(`重命名失败：${rpcError.message}`);
  }
}

function cancelSessionRename() {
  elements.sessionRenameForm.hidden = true;
  elements.sessionTitle.hidden = false;
  elements.sessionTitleInput.disabled = false;
  elements.sessionRenameSave.disabled = false;
}

async function renameSession(event) {
  event.preventDefault();
  const sessionID = selectedSessionID();
  const title = elements.sessionTitleInput.value.trim();
  const error = validateSessionTitle(title);
  if (!sessionID || error) {
    showToast(error || "未选择会话");
    elements.sessionTitleInput.focus();
    return;
  }
  if (title === state.selectedSession.title) {
    cancelSessionRename();
    return;
  }

  elements.sessionTitleInput.disabled = true;
  elements.sessionRenameSave.disabled = true;
  try {
    await saveSessionTitle(sessionID, title);
    cancelSessionRename();
  } catch (error) {
    elements.sessionTitleInput.disabled = false;
    elements.sessionRenameSave.disabled = false;
    elements.sessionTitleInput.focus();
    showToast(`重命名失败：${error.message}`);
  }
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

async function startTurn(userInput) {
  const sessionID = selectedSessionID();
  if (!sessionID || !userInput.trim()) return;
  const text = userInput.trim();
  elements.promptInput.value = "";
  resizePrompt();
  appendMessage("user", text);
  state.liveItems.set(sessionID, new Map());
  state.itemElements.clear();
  state.credentialRequest = null;
  state.busySessions.add(sessionID);
  updateControls();
  renderSessionList();
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

async function switchPermissionProfile(profileID) {
  const sessionID = selectedSessionID();
  if (!sessionID || profileID === state.selectedSession.permission_profile_id) return;
  try {
    await rpc("permission.switch", { session_id: sessionID, profile_id: profileID });
  } catch (error) {
    updateControls();
    showToast(`权限切换失败：${error.message}`);
  }
}

async function switchSandbox(mode) {
  const sessionID = selectedSessionID();
  if (!sessionID) {
    setChoiceMenuValue(
      elements.sandboxMenuTrigger,
      elements.sandboxMenuValue,
      elements.sandboxMenu,
      mode,
      sandboxLabels[mode] || mode,
    );
    return;
  }
  if (mode === state.selectedSession.sandbox_mode) return;
  try {
    await rpc("sandbox.switch", { session_id: sessionID, mode });
  } catch (error) {
    updateControls();
    showToast(`沙箱切换失败：${error.message}`);
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
    const lines = [compactJSON(details.argv), `cwd: ${details.cwd || "."}`];
    if (details.sandbox) {
      lines.push(
        `sandbox: ${details.sandbox.mode}`,
        `write: ${details.sandbox.filesystem_write}; network: ${details.sandbox.network}`,
      );
    }
    elements.approvalDetails.textContent = lines.join("\n");
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
  const recoveryRequired = state.selectedSession?.pending_turn_pause_reason === "recovery_required";
  if (recoveryRequired && !elements.recoveryConfirmed.checked) return;
  try {
    state.busySessions.add(sessionID);
    state.liveItems.set(sessionID, new Map());
    state.itemElements.clear();
    updateControls();
    renderSessionList();
    await rpc(recoveryRequired ? "turn.recover" : "turn.retry", { session_id: sessionID });
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

function applyItemEvent(event, status) {
  const sessionID = event.session_id;
  const payload = event.payload || {};
  const items = liveItemsFor(sessionID);
  const current = items.get(payload.item_id) || {
    item_id: payload.item_id,
    turn_id: event.turn_id,
    item_type: payload.item_type,
    step: Number(payload.step || 0),
    status: "in_progress",
    content: "",
    data: {},
  };
  current.status = status;
  current.data = { ...current.data, ...(payload.data || {}) };
  if (event.type === "item.delta") current.content += payload.delta || "";
  if (event.type === "item.completed") current.content = payload.content || "";
  if (event.type === "item.failed") current.data.error = payload.error || "";
  items.set(current.item_id, current);

  if (current.item_type === "tool_call") {
    const name = current.data.name || "tool";
    if (event.type === "item.started") {
      addActivity(eventLabel(sessionID, `${name} 已提出`), compactJSON(current.data.arguments), "neutral");
    } else if (event.type === "item.delta" && current.data.phase === "running") {
      addActivity(eventLabel(sessionID, `${name} 运行中`), compactJSON(current.data.arguments), "neutral");
    } else if (event.type === "item.completed") {
      const ok = Boolean(current.data.ok);
      const duration = current.data.duration_ms;
      addActivity(
        eventLabel(sessionID, `${name} ${ok ? "已完成" : "失败"}`),
        duration == null ? current.data.status || "completed" : `${current.data.status} · ${duration}ms`,
        ok ? "success" : "danger",
      );
    } else if (event.type === "item.failed") {
      addActivity(eventLabel(sessionID, `${name} 已中断`), current.data.error || "", "warning");
    }
  }
  renderLiveItems(sessionID);
}

async function handleTerminalEvent(event, tone, detail) {
  const sessionID = event.session_id;
  state.busySessions.delete(sessionID);
  if (sessionID === selectedSessionID()) {
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
      loadProjection(result.projection);
      syncCredentialRequest(state.selectedSession);
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
  state.lastSequence.set(sessionID, event.sequence);

  switch (event.type) {
    case "turn.started": {
      state.liveItems.set(sessionID, new Map());
      state.busySessions.add(sessionID);
      addActivity(eventLabel(sessionID, "Turn 已开始"), event.turn_id || "", "neutral");
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
    case "item.started":
      applyItemEvent(event, "in_progress");
      break;
    case "item.delta":
      applyItemEvent(event, "in_progress");
      break;
    case "item.completed":
      applyItemEvent(event, "completed");
      break;
    case "item.failed":
      applyItemEvent(event, payload.status || "failed");
      break;
    case "provider.retry":
      addActivity(
        eventLabel(sessionID, `Provider 重试 ${payload.attempt}`),
        `${payload.delay_seconds}s · ${payload.error}`,
        "warning",
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
    case "turn.continuation_required":
      state.busySessions.delete(sessionID);
      state.attentionSessions.add(sessionID);
      if (sessionID === selectedSessionID()) {
        state.selectedSession.pending_turn_id = event.turn_id;
        state.selectedSession.pending_turn_pause_reason = "step_budget";
        state.selectedSession.pending_turn_pause_details = payload;
        state.liveItems.delete(sessionID);
        updateControls();
      }
      addActivity(
        eventLabel(sessionID, "等待是否继续"),
        `已完成 ${payload.steps_completed} 步`,
        "warning",
      );
      if (sessionID === selectedSessionID()) await selectSession(sessionID);
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
    case "session.renamed":
      updateSessionSummary(sessionID, { title: payload.title });
      if (sessionID === selectedSessionID()) {
        state.selectedSession.title = payload.title;
        updateControls();
      }
      addActivity(eventLabel(sessionID, "会话已重命名"), payload.title, "success");
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
    case "permission.changed":
      if (sessionID === selectedSessionID()) {
        state.selectedSession.permission_profile_id = payload.profile_id;
        updateControls();
      }
      addActivity(eventLabel(sessionID, "权限已切换"), payload.profile_id, "success");
      break;
    case "permission.grants.changed":
      if (sessionID === selectedSessionID()) {
        state.permissionGrants = payload.grants || [];
        renderPermissionGrants();
      }
      addActivity(
        eventLabel(
          sessionID,
          payload.reason === "granted" ? "已添加本次运行授权" : "本次运行授权已更新",
        ),
        `${(payload.grants || []).length} active`,
        "neutral",
      );
      break;
    case "goal.updated":
      if (sessionID === selectedSessionID()) {
        state.currentGoal = payload.goal || null;
        renderGoal();
        refreshContext();
      }
      addActivity(eventLabel(sessionID, "目标进度已更新"), payload.goal?.objective || "", "neutral");
      break;
    case "goal.cleared":
      if (sessionID === selectedSessionID()) {
        state.currentGoal = null;
        renderGoal();
        refreshContext();
      }
      addActivity(eventLabel(sessionID, "目标已清除"), payload.goal_id || "", "warning");
      break;
    case "sandbox.changed":
      if (sessionID === selectedSessionID()) {
        state.selectedSession.sandbox_mode = payload.mode;
        updateControls();
      }
      addActivity(eventLabel(sessionID, "沙箱已切换"), payload.mode, "success");
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
        state.liveItems.delete(sessionID);
        state.credentialRequest = null;
        updateControls();
      }
      addActivity(eventLabel(sessionID, "Turn 已放弃"), payload.turn_id, "warning");
      await selectSession(sessionID);
      break;
    default:
      addActivity(eventLabel(sessionID, event.type), "", "neutral");
  }
  renderSessionList();
  updateControls();
}

async function subscribeEvents({ replay = false } = {}) {
  const params = {};
  if (replay) {
    const active = new Set(state.sessions.filter((session) => session.active).map((session) => session.session_id));
    const cursors = {};
    for (const [sessionID, sequence] of state.lastSequence.entries()) {
      if (active.has(sessionID)) cursors[sessionID] = sequence;
    }
    if (Object.keys(cursors).length) params.after_sequences = cursors;
  }
  const result = await rpc("events.subscribe", params);
  state.subscriptionID = result.subscription_id || "";
  if (!state.subscriptionID) throw new Error("server did not create an event subscription");
}

async function pollEvents() {
  let failures = 0;
  while (!state.stopped) {
    try {
      if (!state.subscriptionID) await subscribeEvents({ replay: true });
      const query = new URLSearchParams({
        timeout: "20",
        subscription_id: state.subscriptionID,
      });
      const response = await fetch(`/api/events?${query}`, {
        headers: { "X-My-Agent-Token": bootstrap.token },
        cache: "no-store",
      });
      verifyServerInstance(response);
      if (response.status === 204) {
        setConnection("已连接", "online");
        failures = 0;
        continue;
      }
      if (response.status === 410) {
        state.subscriptionID = "";
        await subscribeEvents({ replay: true });
        if (selectedSessionID()) await selectSession(selectedSessionID());
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
  elements.sessionTitle.addEventListener("click", beginSessionRename);
  elements.sessionRenameForm.addEventListener("submit", renameSession);
  elements.sessionRenameCancel.addEventListener("click", cancelSessionRename);
  elements.sessionTitleInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      cancelSessionRename();
      elements.sessionTitle.focus();
    }
  });
  elements.sessionMenuRename.addEventListener("click", () => {
    const sessionID = state.sessionMenuSessionID;
    if (sessionID) beginSidebarRename(sessionID);
  });
  elements.sessionContextMenu.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      const sessionID = state.sessionMenuSessionID;
      closeSessionMenu();
      elements.sessionList
        .querySelector(`.session-row[data-session-id="${sessionID}"] .session-actions`)
        ?.focus();
    }
  });
  bindChoiceMenu(elements.modelMenuTrigger, elements.modelMenu);
  bindChoiceMenu(elements.permissionMenuTrigger, elements.permissionMenu);
  bindChoiceMenu(elements.sandboxMenuTrigger, elements.sandboxMenu);
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".choice-menu")) closeChoiceMenus();
    if (
      !event.target.closest("#session-context-menu") &&
      !event.target.closest(".session-actions")
    ) {
      closeSessionMenu();
    }
  });
  elements.sessionList.addEventListener("scroll", closeSessionMenu);
  window.addEventListener("resize", closeSessionMenu);
  elements.stopTurn.addEventListener("click", interruptTurn);
  elements.executePlan.addEventListener("click", executePlan);
  elements.revisePlan.addEventListener("click", revisePlan);
  elements.retryTurn.addEventListener("click", retryTurn);
  elements.recoveryConfirmed.addEventListener("change", updateControls);
  elements.abortTurn.addEventListener("click", abortTurn);
  elements.credentialPanel.addEventListener("submit", saveCredential);
  elements.credentialAbort.addEventListener("click", abortTurn);
  elements.clearActivity.addEventListener("click", () => {
    state.activities = [];
    renderActivities();
  });
  elements.clearPermissionGrants.addEventListener("click", clearPermissionGrants);
  elements.clearGoal.addEventListener("click", clearGoal);
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
  if (!bootstrap?.token || !bootstrap?.instanceID) {
    setConnection("初始化失败", "offline");
    showToast("Agent 服务版本已过期，请重启服务并刷新页面");
    return;
  }
  bindUI();
  resetContext();
  renderGoal();
  renderPermissionGrants();
  try {
    const [models, modes, permissions, sandboxes, sessions] = await Promise.all([
      rpc("model.list"),
      rpc("mode.list"),
      rpc("permission.list"),
      rpc("sandbox.list"),
      rpc("session.list"),
    ]);
    state.models = models.models || [];
    state.credentialsPath = models.credentials_path || "";
    state.modes = modes.modes || [];
    state.permissionProfiles = permissions.profiles || [];
    state.sandboxProfiles = sandboxes.profiles || [];
    state.sessions = sessions.sessions || [];
    renderModelCatalog();
    renderModeCatalog();
    renderPermissionCatalog();
    renderSandboxCatalog();
    renderSessionList();
    updateControls();
    await subscribeEvents();
    pollEvents();
    setConnection("已连接", "online");
    if (state.sessions.length) await selectSession(state.sessions[0].session_id);
  } catch (error) {
    setConnection("初始化失败", "offline");
    showToast(`初始化失败：${error.message}`);
  }
}

initialize();
