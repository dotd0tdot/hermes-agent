# Errors

Command failures and integration errors.

---

## [ERR-20260623-001] mc_server_config_phase

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
MC server client disconnects with "Missing registry: dimension_type" during CONFIG phase

### Error
```
Failed to decode packet 'clientbound/minecraft:login'
Missing registry: ResourceKey[minecraft:root / minecraft:dimension_type]
```

### Context
- Command/operation attempted: RustMC server accepting MC 1.21.4 client connection
- The server sent registry_data before handling client's known_packs request
- CONFIG phase was implemented as single-shot, not as packet loop
- Fix: loop-based CONFIG phase, respond to known_packs (0x08 S→C) before sending registry_data

### Suggested Fix
Implement CONFIG phase as packet processing loop with timeout. Send registry_data only after known_packs exchange.

### Resolution
- **Resolved**: 2026-06-01
- **Notes**: Rewrote CONFIG phase to loop on client packets, handle known_packs, then send registry_data. Also fixed NBT format to use write_blob.

### Metadata
- Reproducible: yes
- Related Files: /home/dot/minecraft/rustmc/crates/mc-network/src/server.rs
- See Also: LRN-20260623-001, LRN-20260623-007

---

## [ERR-20260623-002] scanner_semaphore_deadlock

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
ScanEngine progress stuck at 0/N — submitter thread blocked on Semaphore.acquire()

### Error
Progress always shows "0 / N" and never advances. Tasks are submitted but never execute.

### Context
- Command/operation attempted: Server Scanner mod scan on NeoForge 1.21.5
- Root cause: stopScan() called rateLimiter.drainPermits() which consumed all permits
- startScan() then tried acquire() on the same drained Semaphore → permanent block
- Fix: create fresh Semaphore(maxThreads*2) in startScan(), discard old executor

### Suggested Fix
Never reuse a Semaphore after drainPermits(). Create fresh instances in start patterns.

### Resolution
- **Resolved**: 2026-06-12
- **Notes**: Rewrote ScanEngine.startScan() to create new Semaphore and ThreadPoolExecutor. Verified with smoke test: 3 tasks complete, progress advances.

### Metadata
- Reproducible: yes
- Related Files: /home/dot/server-scanner/src/main/java/org/dot/serverscanner/scanner/ScanEngine.java
- See Also: LRN-20260623-002

---

## [ERR-20260623-003] scanner_proxy_timeout

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
All SLP pings timeout because sockets route through SOCKS proxy

### Error
All scan results show TIMEOUT. debug.log shows `SocksSocketImpl.connect` in call stack.

### Context
- Gradle proxy settings (for dependency downloads) leaked into JVM-wide proxy
- All Socket connections from the mod went through SOCKS proxy
- LAN addresses (192.168.x.x) unreachable through external proxy
- Fix: set Proxy.NO_PROXY explicitly on Socket creation

### Suggested Fix
Always use Proxy.NO_PROXY for custom socket connections in mods.

### Resolution
- **Resolved**: 2026-06-12
- **Notes**: Updated SLPProtocol.java to use new Socket(Proxy.NO_PROXY). Also user cleaned up Gradle proxy settings.

### Metadata
- Reproducible: yes
- Related Files: /home/dot/server-scanner/src/main/java/org/dot/serverscanner/scanner/SLPProtocol.java

---

## [ERR-20260623-004] zerkalo_registry_frozen

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
ZerkaloDataFixer crashes with "Registry is already frozen" when registering feature aliases

### Error
```
java.lang.IllegalStateException: Registry is already frozen
```

### Context
- Attempted to register removed Minecraft feature aliases (random_patch, flower, forest_rock) into frozen registry
- Direct MappedRegistry.register() fails after bootstrap phase
- Fix: use NeoForge RegisterEvent listener (primary), reflection-based unfreeze (fallback)

### Suggested Fix
Always use NeoForge event system for registry operations. Reflection fallback only for extreme cases.

### Resolution
- **Resolved**: 2026-06-14
- **Notes**: Implemented dual-path registration: RegisterEvent listener + reflection unfreeze fallback. Tested with Terralith 1.21.1 JAR.

### Metadata
- Reproducible: yes
- Related Files: /run/media/dot/3da5c032-623d-49ab-9f1a-afe455ed1613/build/zerkalo/runtime/src/main/java/io/zerkalo/runtime/ZerkaloDataFixer.java
- See Also: LRN-20260623-006

---
