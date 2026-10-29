// Requires Playwright on NODE_PATH; the fixture never calls a model API.
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { mkdirSync } = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const child = spawn(path.join(root, '.venv/bin/python'), ['tests/web_fixture.py'], {
    cwd: root, env: { ...process.env, PYTHONPATH: root }, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let browser;
  let page;
  let stderr = '';
  child.stderr.on('data', data => { stderr += data; });
  try {
    const url = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Fixture startup timeout: ${stderr}`)), 10000);
      child.stdout.once('data', data => { clearTimeout(timer); resolve(data.toString().trim()); });
      child.once('exit', code => { clearTimeout(timer); reject(new Error(`Fixture exited ${code}: ${stderr}`)); });
    });
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    const errors = [];
    const requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.url().endsWith('/api/rpc')) requests.push(request.postDataJSON());
    });
    await page.goto(url);
    await page.waitForFunction(() => state.selectedSession && !elements.promptInput.disabled);
    for (let i = 0; i < 3; i++) await page.locator('#new-session').click();
    assert.equal(requests.filter(r => r.method === 'session.create').length, 0);
    assert.equal(await page.locator('.session-item').count(), 0);
    await page.locator('#prompt-input').fill('Unsent draft');
    await page.locator('#new-session').click();
    assert.equal(await page.locator('#prompt-input').inputValue(), 'Unsent draft');
    await page.reload();
    await page.waitForFunction(() => state.selectedSession && !elements.promptInput.disabled);
    assert.equal(await page.locator('.session-item').count(), 0);
    await page.locator('#prompt-input').fill('Fix the sample failing test');
    await page.evaluate(() => { startTurn(elements.promptInput.value); startTurn(elements.promptInput.value); });
    await page.waitForFunction(() => !state.sending && state.selectedSession?.message_count === 2);
    assert.equal(requests.filter(r => r.method === 'session.create').length, 1);
    assert.equal(requests.filter(r => r.method === 'turn.start').length, 1);
    assert.equal(await page.locator('#session-title-text').innerText(), 'Fix the sample failing test');
    const savedID = await page.evaluate(() => selectedSessionID());
    await page.reload();
    await page.waitForFunction(() => state.selectedSession?.message_count === 2);
    assert.equal(await page.evaluate(() => selectedSessionID()), savedID);
    await page.locator('#new-session').click();
    await page.locator('#session-title').click();
    await page.locator('#session-title-input').fill('Manual draft title');
    await page.locator('#session-title-input').press('Enter');
    assert.equal(requests.filter(r => r.method === 'session.create').length, 1);
    await page.locator('#prompt-input').fill('Keep my title');
    await page.locator('#send-prompt').click();
    await page.waitForFunction(() => !state.sending && state.selectedSession?.message_count === 2);
    assert.equal(await page.locator('#session-title-text').innerText(), 'Manual draft title');
    await page.locator('#new-session').click();
    await page.locator('.mode-button[data-mode-id="plan"]').click();
    assert.equal(requests.filter(r => r.method === 'mode.switch').length, 0);
    await page.locator('.mode-button[data-mode-id="build"]').click();
    await page.locator('#prompt-input').fill('Retry after a lost creation response');
    const beforeRetry = requests.length;
    let createAttempts = 0;
    await page.route('**/api/rpc', async route => {
      if (route.request().postDataJSON().method !== 'session.create') return route.continue();
      createAttempts += 1;
      if (createAttempts === 1) await route.fetch();
      if (createAttempts <= 2) return route.abort('failed');
      return route.continue();
    });
    await page.locator('#send-prompt').click();
    await page.waitForFunction(() => !state.sending && state.draftCreation !== null);
    assert.equal(await page.locator('#prompt-input').inputValue(), 'Retry after a lost creation response');
    await page.locator('#send-prompt').click();
    await page.waitForFunction(() => !state.sending && state.selectedSession?.message_count === 2);
    const creations = requests.slice(beforeRetry).filter(r => r.method === 'session.create');
    assert.equal(creations.length, 3);
    assert.equal(new Set(creations.map(r => r.request_id)).size, 1);
    assert.equal(await page.evaluate(async () => (await rpc('session.list')).sessions.length), 3);
    assert.match(await page.locator('#verification-view').innerText(), /暂无修改或验证记录/);
    // Rendering check only: authoritative projection behavior is covered by Python tests.
    await page.evaluate(() => renderVerification({
      turn_id: `${selectedSessionID()}:1`, status: 'stale',
      changed_files: [{ path: 'src/a-very-long-workspace-relative-filename-for-wrapping.py' }],
      checks: [{ argv: ['python', '-m', 'unittest', 'discover', '-s', 'tests'], cwd: '.',
        status: 'passed', exit_code: 0, finished_at: new Date().toISOString(), freshness: 'stale' }],
      unknown_attempts: [],
    }));
    assert.match(await page.locator('#verification-view').innerText(), /检查后又发生修改/);
    const out = path.join(root, '.my-agent/ui-checks');
    mkdirSync(out, { recursive: true });
    await page.screenshot({ path: path.join(out, 'desktop.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#inspector-toggle').click();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'mobile.png') });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ passed: true, realModelCalls: 0, checks: ['draft', 'double-send', 'title', 'reload', 'same-id-retry', 'verification', 'mobile'], screenshots: out }));
  } catch (error) {
    if (page) console.error(await page.evaluate(() => ({ session: state.selectedSession, toast: elements.toast.textContent, activities: state.activities })));
    console.error(stderr);
    throw error;
  } finally {
    if (browser) await browser.close();
    child.kill('SIGTERM');
    await new Promise(resolve => child.exitCode !== null || child.signalCode !== null ? resolve() : child.once('exit', resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
