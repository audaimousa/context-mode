import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { mkdtempSync, readFileSync, writeFileSync, chmodSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { HermesAdapter } from "../../src/adapters/hermes/index.js";

const ROOT = resolve(__dirname, "../..");

describe("Hermes native adapter", () => {
  it("declares only documented Hermes hooks", () => {
    const manifest = readFileSync(resolve(ROOT, "plugin.yaml"), "utf8");
    expect(manifest).toContain("provides_hooks:");
    expect(manifest).toContain("- on_session_start");
    expect(manifest).toContain("- transform_tool_result");
    expect(manifest).not.toContain("pre_llm_call");
    expect(manifest).not.toMatch(/^hooks:/m);
  });

  it("uses profile-scoped storage and current public capabilities", () => {
    const previous = process.env.HERMES_HOME;
    const home = mkdtempSync(resolve(tmpdir(), "hermes-home-"));
    process.env.HERMES_HOME = home;
    try {
      const adapter = new HermesAdapter();
      expect(adapter.getConfigDir()).toBe(home);
      expect(adapter.getSettingsPath()).toBe(resolve(home, "config.yaml"));
      expect(adapter.getSessionDir()).toBe(resolve(home, "context-mode", "sessions"));
      expect(adapter.parsePreToolUseInput({
        tool_name: "terminal", args: { command: "pwd" }, session_id: "s",
      })).toMatchObject({
        toolName: "terminal", toolInput: { command: "pwd" }, sessionId: "s",
      });
      expect(adapter.capabilities.canModifyArgs).toBe(true);
      expect(adapter.capabilities.canModifyOutput).toBe(true);
      expect(adapter.capabilities.canInjectSessionContext).toBe(false);
      expect(adapter.paradigm).toBe("python-plugin");
    } finally {
      if (previous === undefined) delete process.env.HERMES_HOME;
      else process.env.HERMES_HOME = previous;
    }
  });

  it("registers public hooks, preserves Hermes execution, and fails open", () => {
    // The Docker test backend mounts /tmp noexec; place the executable stub
    // under the checkout and remove it before asserting.
    const binDir = mkdtempSync(resolve(ROOT, ".hermes-test-"));
    const stubPy = resolve(binDir, "stub.py");
    writeFileSync(stubPy, String.raw`import os, sys
sys.stdin.read()
print(os.environ.get("HERMES_STUB_RESPONSE", "{}"), end="")
`);
    const windows = process.platform === "win32";
    const stub = resolve(binDir, windows ? "context-mode.cmd" : "context-mode");
    writeFileSync(stub, windows
      ? `@python "${stubPy}" %*\r\n`
      : `#!/bin/sh\nexec python3 "${stubPy}" "$@"\n`);
    if (!windows) chmodSync(stub, 0o755);

    const harness = String.raw`
import importlib.util, json, os, pathlib, tempfile, time
root=pathlib.Path(${JSON.stringify(ROOT)})
spec=importlib.util.spec_from_file_location("context_mode_hermes", root/"__init__.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class C:
 def __init__(self): self.hooks={}; self.commands={}; self.calls=[]; self.delay=False; self.unload=None
 def register_hook(self,n,f): self.hooks[n]=f
 def register_command(self,n,f,*a): self.commands[n]=f
 def on_unload(self,f): self.unload=f
 def dispatch_tool(self,n,a,**kw):
  self.calls.append((n,a,kw))
  if self.delay: time.sleep(0.2)
  return json.dumps({"result":{"content":[{"type":"text","text":"Indexed 1 chunk"}]}})
c=C(); m.register(c)
assert {"pre_tool_call","post_tool_call","on_session_start","on_session_end","on_session_finalize","transform_tool_result"} == set(c.hooks)
assert {"ctx-stats","ctx-doctor","ctx-search"} == set(c.commands)
assert not hasattr(m, "_pre_llm_call")
os.environ["HERMES_STUB_RESPONSE"] = json.dumps({"action":"modify","args":{"command":"echo guidance"}})
r=c.hooks["pre_tool_call"]("terminal", {"command":"curl https://example.com"}, session_id="s")
assert r == {"action":"modify","args":{"command":"echo guidance"}}, repr(r)
assert not c.calls
os.environ["HERMES_STUB_RESPONSE"] = json.dumps({"action":"approve","message":"confirm"})
assert c.hooks["pre_tool_call"]("terminal", {"command":"danger"}) == {"action":"approve","message":"confirm"}
os.environ["HERMES_STUB_RESPONSE"] = "not-json"
assert c.hooks["pre_tool_call"]("terminal", {"command":"unchanged"}) is None
large="x"*17000
assert c.hooks["transform_tool_result"]("terminal", large, session_id="s") is None
marker=c.hooks["transform_tool_result"]("read_file", large, session_id="s", tool_call_id="call-a")
assert marker and "indexed 17000 bytes" in marker
assert c.calls[-1][0] == "mcp__context_mode__ctx_index"
assert c.hooks["transform_tool_result"]("mcp__context_mode__ctx_search", large, session_id="s") is None
original_dispatch=c.dispatch_tool
def fail(*a, **k): raise RuntimeError("offline")
c.dispatch_tool=fail
assert c.hooks["transform_tool_result"]("read_file", large, session_id="s") is None
c.dispatch_tool=original_dispatch
c.delay=True; m._INDEX_TIMEOUT=0.01
started=time.monotonic()
assert c.hooks["transform_tool_result"]("read_file", large, session_id="s") is None
assert time.monotonic()-started < 0.1
assert all(call[0] != "terminal" for call in c.calls)
print("ok")
`;
    const run = spawnSync("python3", ["-c", harness], {
      encoding: "utf8",
      timeout: 10_000,
      env: {
        ...process.env,
        CONTEXT_MODE_EXECUTABLE: stub,
        HERMES_HOME: resolve(binDir, "profile"),
      },
    });
    rmSync(binDir, { recursive: true, force: true });
    expect(run.status, run.stderr).toBe(0);
    expect(run.stdout.trim()).toBe("ok");
  });
});
