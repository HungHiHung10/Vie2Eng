const { test, before } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

let clientSource;
const dependencySources = new Map();
before(async () => {
  if (process.env.VIE2ENG_TEST_URL) {
    const base = process.env.VIE2ENG_TEST_URL;
    const html = await (await fetch(base)).text();
    const script = html.match(/<script\b[^>]*src="([^"]*app\.js[^\"]*)"/)[1];
    assert.match(script, /\?v=/, 'The page must request a versioned client script');
    clientSource = await (await fetch(new URL(script, base))).text();
    assert.equal(clientSource, fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8'));
    for (const filename of fs.readdirSync(path.join(__dirname, '../static/js'))) {
      if (!filename.endsWith('.js')) continue;
      const source = await (await fetch(new URL('/static/js/' + filename, base))).text();
      const filePath = path.resolve(__dirname, '../static/js', filename);
      assert.equal(source, fs.readFileSync(filePath, 'utf8'));
      dependencySources.set(filePath, source);
    }
  } else {
    clientSource = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
  }
});

async function createApp(records = {}) {
  const elements = new Map();
  const storage = new Map(Object.entries(records).map(([key, value]) => [key, JSON.stringify(value)]));
  const requests = [];
  const responses = [];
  const spoken = [];
  const timers = new Map();
  let nextTimer = 0;
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      const attrs = new Map();
      elements.set(id, {
        value: '', textContent: '', style: {}, listeners: {},
        classList: {
          add: (...names) => names.forEach(name => classes.add(name)),
          remove: (...names) => names.forEach(name => classes.delete(name)),
          contains: name => classes.has(name),
          toggle(name, force) {
            const active = force === undefined ? !classes.has(name) : force;
            active ? classes.add(name) : classes.delete(name);
            return active;
          },
        },
        addEventListener(name, fn) { this.listeners[name] = fn; },
        setAttribute(name, value) { attrs.set(name, value); },
        getAttribute(name) { return name === 'data-idx' ? '0' : attrs.get(name); },
        querySelectorAll: () => [element(id + '-record')],
        focus() {},
      });
    }
    return elements.get(id);
  }
  const context = {
    document: {
      addEventListener(name, fn) { if (name === 'DOMContentLoaded') fn(); },
      getElementById: element, querySelectorAll: () => [],
      createElement: () => ({ textContent: '', get innerHTML() { return this.textContent; } }),
      documentElement: element('root'),
    },
    localStorage: {
      getItem: key => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    window: { speechSynthesis: { cancel() {}, speak: utterance => spoken.push(utterance) } },
    SpeechSynthesisUtterance: function (text) { this.text = text; },
    setTimeout(fn) { timers.set(++nextTimer, fn); return nextTimer; },
    clearTimeout: id => timers.delete(id),
    console,
    fetch: async (_, options) => {
      requests.push(JSON.parse(options.body));
      assert.ok(responses.length, 'Unexpected extra translation request');
      const data = await responses.shift();
      return { ok: true, json: async () => data };
    },
  };
  const moduleContext = vm.createContext(context);
  const modules = new Map();
  const entryPath = path.resolve(__dirname, '../static/app.js');
  async function loadModule(identifier) {
    if (modules.has(identifier)) return modules.get(identifier);
    const source = identifier === entryPath ? clientSource
      : dependencySources.get(identifier) ?? fs.readFileSync(identifier, 'utf8');
    const module = new vm.SourceTextModule(source, { context: moduleContext, identifier });
    modules.set(identifier, module);
    await module.link((specifier, parent) => loadModule(path.resolve(path.dirname(parent.identifier), specifier)));
    return module;
  }
  await (await loadModule(entryPath)).evaluate();
  return {
    element, requests, responses, spoken,
    click: id => element(id).listeners.click(),
    flush: () => new Promise(resolve => setImmediate(resolve)),
    assertTarget(lang) {
      assert.equal(element('tgtViBtn').classList.contains('active'), lang === 'vi');
      assert.equal(element('tgtEnBtn').classList.contains('active'), lang === 'en');
    },
  };
}

function translation(target, fromCache = false) {
  return {
    translated_text: target === 'vi' ? 'Tôi đang học' : 'I am learning',
    detected_lang: target === 'vi' ? 'en' : 'vi',
    source_lang: target === 'vi' ? 'en' : 'vi', target_lang: target,
    model_name: 'HungNguyen29/MarianNMT-en2vi-ct2-int8',
    latency_ms: 0, from_cache: fromCache,
  };
}

test('auto detection updates target, speech and next request, including cached responses', async () => {
  const app = await createApp();
  app.element('sourceText').value = 'im am learning';
  for (const target of ['vi', 'en']) {
    app.responses.push(translation(target, true));
    app.click('translateBtn');
    await app.flush();
    app.assertTarget(target);
    app.click('listenTargetBtn');
    assert.equal(app.spoken.at(-1).lang, target === 'vi' ? 'vi-VN' : 'en-US');
  }
  assert.equal(app.requests[1].target_lang, 'vi');
});

for (const [key, button, list] of [
  ['phomt_history', 'historyModalBtn', 'historyList'],
  ['phomt_saved', 'savedModalBtn', 'savedList'],
]) {
  test(`${key} restores the language selectors and speech without a translation request`, async () => {
    const app = await createApp({ [key]: [{ src: 'im am learning', tgt: 'Tôi đang học', srcLang: 'auto', tgtLang: 'vi' }] });
    app.click(button);
    app.click(list + '-record');
    app.assertTarget('vi');
    assert.equal(app.element('srcAutoBtn').classList.contains('active'), true);
    app.click('listenTargetBtn');
    assert.equal(app.spoken[0].lang, 'vi-VN');
    assert.equal(app.requests.length, 0);
  });
}

test('swapping languages sends one request with the swapped text and languages', async () => {
  const app = await createApp();
  app.element('sourceText').value = 'im am learning';
  app.responses.push(translation('vi'));
  app.click('translateBtn');
  await app.flush();
  app.responses.push(translation('en'));
  app.click('swapLangBtn');
  await app.flush();
  assert.equal(app.requests.length, 2);
  assert.equal(app.requests[1].text, 'Tôi đang học');
  assert.equal(app.requests[1].source_lang, 'vi');
  assert.equal(app.requests[1].target_lang, 'en');
  app.assertTarget('en');
});

test('a late translation cannot overwrite a restored history item', async () => {
  const app = await createApp({ phomt_history: [{ src: 'Xin chào', tgt: 'Hello', srcLang: 'vi', tgtLang: 'en' }] });
  let finish;
  app.responses.push(new Promise(resolve => { finish = resolve; }));
  app.element('sourceText').value = 'im am learning';
  app.click('translateBtn');
  app.click('historyModalBtn');
  app.click('historyList-record');
  finish(translation('vi'));
  await app.flush();
  app.assertTarget('en');
  assert.equal(app.element('targetResult').textContent, 'Hello');
});

test('editing input invalidates an older response during the typing delay', async () => {
  const app = await createApp();
  let finish;
  app.responses.push(new Promise(resolve => { finish = resolve; }));
  app.element('sourceText').value = 'im am learning';
  app.click('translateBtn');
  app.element('sourceText').value = 'Xin chào';
  app.element('sourceText').listeners.input();
  finish(translation('vi'));
  await app.flush();
  app.assertTarget('en');
  assert.equal(app.element('targetResult').textContent, '');
});
