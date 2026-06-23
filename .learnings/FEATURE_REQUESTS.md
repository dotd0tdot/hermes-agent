# Feature Requests

Capabilities requested by the user.

---

## [FEAT-20260623-001] zerkalo_data_version_shim

**Logged**: 2026-06-23T20:50:00Z
**Priority**: high
**Status**: in_progress
**Area**: backend

### Requested Capability
Emulate MC 1.21.1 data-pack schema behavior so Terralith 1.21.1 works on NeoForge 26.1.2 without patching the Minecraft client

### User Context
Terralith 1.21.1 uses old JSON schemas (carvers, state_providers, wolf variants, feature types) that MC 26.1.2 no longer understands. User wants to avoid patching the Minecraft client JAR.

### Complexity Estimate
complex

### Suggested Implementation
ZerkaloDataFixer v2: JSON schema translation layer that rewrites Terralith JAR data entries before NeoForge parses them. Covers carvers, state_providers, wolf variants, huge mushrooms, unnamespaced types.

### Metadata
- Frequency: first_time
- Related Features: ZerkaloDataFixer, ZerkaloServiceLocator

---

## [FEAT-20260623-002] server_scanner_improvements

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: pending
**Area**: frontend

### Requested Capability
Server Scanner mod: CSV export, double-click connect, hover highlight, scroll support

### User Context
Initial release of the scanner had basic functionality. User wanted better UX: row hover, double-click to join, scroll wheel support, CSV export.

### Complexity Estimate
medium

### Suggested Implementation
Already implemented in ScannerScreen.java rewrite. Remaining: CSV export file chooser, favorites/history persistence.

### Metadata
- Frequency: first_time
- Related Features: ScannerScreen, ScanEngine

---

## [FEAT-20260623-003] rustmc_protocol_debugging

**Logged**: 2026-06-23T20:50:00Z
**Priority**: medium
**Status**: pending
**Area**: docs

### Requested Capability
RustMC server protocol debugging skill: packet IDs, CONFIG flow, typical errors

### User Context
Debugging MC 1.21.4 (v767) protocol in Rust required extensive trial-and-error. A skill would speed up future protocol work.

### Complexity Estimate
simple

### Suggested Implementation
Create skill with: packet ID table, CONFIG→LOGIN→PLAY flow, NBT format rules, common errors and fixes.

### Metadata
- Frequency: recurring
- Related Features: RustMC, mc-network

---
