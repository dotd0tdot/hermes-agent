# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260623-001] best_practice

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
Minecraft CONFIG phase is a packet loop — don't send registry_data before handling known_packs

### Details
When implementing the RustMC server for MC 1.21.4 (v767), the client disconnects with "Missing registry: dimension_type" after the CONFIG phase. Root cause: the server sent registry_data and immediately sent finish_configuration, but the client first sends known_packs (0x07 C→S) which must be responded to before registry_data is sent. The CONFIG phase is a state machine: the server must loop reading client packets (known_packs, cookie requests, custom payloads) and only send registry_data after responding to known_packs or on timeout.

### Suggested Action
Always implement CONFIG phase as a packet processing loop with timeout fallback, never as a single-shot send.

### Metadata
- Source: error
- Related Files: /home/dot/minecraft/rustmc/crates/mc-network/src/server.rs
- Tags: minecraft, protocol, config-phase, rustmc
- Pattern-Key: mc.config_phase_loop
- Recurrence-Count: 2
- First-Seen: 2026-06-01
- Last-Seen: 2026-06-01

---

## [LRN-20260623-002] best_practice

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
Semaphore.drainPermits() in a reusable pool causes permanent deadlock — create fresh instance per scan

### Details
In the Server Scanner mod's ScanEngine, stopScan() called rateLimiter.drainPermits() which consumed all permits to 0. The submitter thread then blocked forever on acquire(). Creating a fresh Semaphore(maxThreads*2) in startScan() solves this: old executor and semaphore are discarded, new ones created. This is the correct pattern for "stop and restart" resource pools.

### Suggested Action
When implementing stop/restart patterns with Semaphores or similar counting primitives, create fresh instances instead of trying to reset state.

### Metadata
- Source: error
- Related Files: /home/dot/server-scanner/src/main/java/org/dot/serverscanner/scanner/ScanEngine.java
- Tags: java, concurrency, semaphore, scanner
- Pattern-Key: concurrency.fresh_semaphore
- Recurrence-Count: 1
- First-Seen: 2026-06-12

---

## [LRN-20260623-003] insight

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
SOCKS proxy from JVM args can leak into runtime socket connections

### Details
Gradle proxy settings (configured for downloading dependencies) sometimes propagate JVM-wide. When the Minecraft client inherits these JVM args, all Socket connections (including from mods) go through the SOCKS proxy, causing timeouts on LAN scans. The fix is to use Proxy.NO_PROXY explicitly in the mod's socket code, or clean up JVM args.

### Suggested Action
Always set Proxy.NO_PROXY on custom Socket connections in mods. Check Gradle properties for proxy settings that might leak.

### Metadata
- Source: error
- Related Files: /home/dot/server-scanner/src/main/java/org/dot/serverscanner/scanner/SLPProtocol.java
- Tags: java, proxy, gradle, networking
- Pattern-Key: infra.proxy_leak

---

## [LRN-20260623-004] best_practice

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
Per-field try/catch in JSON parsing prevents blanket nullification of all fields

### Details
In ServerPinger.java, a blanket `catch(Exception)` around the entire JSON parsing block meant that if ANY field failed to parse, ALL fields became null/empty. The fix: wrap each field extraction in its own try/catch so one bad field doesn't destroy the entire result. This is especially important for Minecraft SLP responses where server JSON is unstandardized.

### Suggested Action
When parsing external/untrusted JSON (server list pings, API responses), use per-field error isolation instead of blanket catch.

### Metadata
- Source: error
- Related Files: /home/dot/server-scanner/src/main/java/org/dot/serverscanner/scanner/ServerPinger.java
- Tags: java, json, error-handling, resilience
- Pattern-Key: resilience.per_field_parse

---

## [LRN-20260623-005] knowledge_gap

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: pending
**Area**: backend

### Summary
MC protocol packet IDs are version-specific — always verify against wiki.vg for exact version

### Details
Multiple iterations of guessing Player Abilities packet ID (0x36, 0x37, 0x38, 0x3A) wasted significant time. The correct approach is to look up the exact packet ID for the specific protocol version on wiki.vg. For v767: Join Game=0x3A, Respawn=0x41, Player Abilities=0x3B, Set Held Item=0x4D. Guessing leads to cascading decode errors where the client reads the wrong packet type.

### Suggested Action
Before implementing any protocol handler, create a mapping table from wiki.vg. Don't guess packet IDs.

### Metadata
- Source: error
- Tags: minecraft, protocol, packet-ids
- Pattern-Key: mc.protocol_packet_ids
- Recurrence-Count: 3
- First-Seen: 2026-05-31
- Last-Seen: 2026-06-01

---

## [LRN-20260623-006] best_practice

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
NeoForge registry is frozen after Bootstrap — use RegisterEvent, not direct MappedRegistry.register()

### Details
In ZerkaloDataFixer, registering feature aliases into a frozen registry throws "Registry is already frozen". The fix has two layers: (1) primary path uses NeoForge RegisterEvent listener (which fires at the right lifecycle stage), (2) fallback uses reflection to temporarily unfreeze MappedRegistry via the private `frozen` field. The RegisterEvent path should always be preferred.

### Suggested Action
Always register new entries via NeoForge event system (RegisterEvent), never bypass it with direct registry manipulation unless as a last resort.

### Metadata
- Source: error
- Related Files: /run/media/dot/3da5c032-623d-49ab-9f1a-afe455ed1613/build/zerkalo/runtime/src/main/java/io/zerkalo/runtime/ZerkaloDataFixer.java
- Tags: neoforge, registry, modding
- Pattern-Key: neoforge.registry_frozen
- Recurrence-Count: 2
- First-Seen: 2026-06-14
- Last-Seen: 2026-06-15

---

## [LRN-20260623-007] insight

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: pending
**Area**: backend

### Summary
NBT in MC protocol must use write_blob format (type_id + name_len + name + content + TAG_END)

### Details
When writing NBT data for registry_data packets, using raw compound tag bytes without the named tag wrapper causes "larger than expected" errors. The correct format is: tag_id(1) + name_len(2) + name(n) + content + TAG_END(0x00). The write_blob() function in mc_core handles this correctly.

### Suggested Action
Always use write_blob() or equivalent named-tag serialization for MC protocol NBT, never raw compound bytes.

### Metadata
- Source: error
- Related Files: /home/dot/minecraft/rustmc/crates/mc-network/src/server.rs
- Tags: minecraft, nbt, protocol, rust
- Pattern-Key: mc.nbt_write_blob
- Recurrence-Count: 2
- First-Seen: 2026-06-01

---

## [LRN-20260623-008] correction

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: pending
**Area**: backend

### Summary
Cloudflare Spectrum / TCPShield servers won't respond to SLP — this is by design, not a bug

### Details
Servers behind Cloudflare Spectrum or TCPShield DDoS protection (like Hypixel on port 255642) close the TCP connection after the SLP handshake without returning a status response. This is intentional DDoS mitigation. The scanner should show these as "PROTECTED" rather than "ERROR" to avoid confusing users.

### Suggested Action
Display "PROTECTED" (yellow) for ERROR/REFUSED/TIMEOUT results in the scanner UI. Document this limitation.

### Metadata
- Source: conversation
- Tags: minecraft, slp, ddos-protection, scanner
- Pattern-Key: mc.slp_cloudflare

---
