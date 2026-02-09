# TruthGPT Model Builder - Refactoring Plan

## 🎯 Objectives

1. **Consolidate duplicate files** - Merge multiple TruthGPT service implementations
2. **Organize into modules** - Move root-level files into proper module structure
3. **Fix missing exports** - Add missing `createTruthGPTModel` function
4. **Update imports** - Ensure all imports use the new modular structure
5. **Clean up root level** - Remove files that should be in modules

## 📊 Current Issues

### 1. Duplicate/Similar Files
- `truthgpt-service.ts` - Main service (missing `createTruthGPTModel`)
- `truthgpt-api-client.ts` - API client
- `truthgpt-api-client-enhanced.ts` - Enhanced API client
- `truthgpt-adapter.ts` - Adapter
- `truthgpt-integrator.ts` - Integrator

### 2. Root-Level Files That Should Be in Modules
- Model-related: `model-analyzer.ts`, `model-manager.ts`, `model-optimizer.ts`, `model-validator.ts`, `model-templates.ts`, `model-versioning.ts`, `model-exporter.ts`
- Service-related: `github-service.ts` (should be in modules/github)
- Utility files: Many utility files at root that should be organized

### 3. Missing Functions
- `createTruthGPTModel` - Referenced but not exported from `truthgpt-service.ts`

## 🏗️ Refactoring Strategy

### Phase 1: Create Missing Functions
1. Add `createTruthGPTModel` to `truthgpt-service.ts` or create it in `services/model-creation-service.ts`
2. Export it properly

### Phase 2: Consolidate TruthGPT Services
1. Review all TruthGPT-related files
2. Consolidate into a single service or clearly separate concerns
3. Update imports

### Phase 3: Organize Files into Modules
1. Move model-related files to `modules/management/` or appropriate module
2. Move `github-service.ts` to `modules/github/`
3. Organize hooks into `hooks/` directory (already exists but may need cleanup)

### Phase 4: Update Imports
1. Update all imports to use new module structure
2. Update module index files
3. Verify no broken imports

### Phase 5: Testing & Verification
1. Run tests
2. Fix any broken imports
3. Verify functionality

## 📁 Target Structure

```
lib/
├── core/                    # Core types and config (already exists)
│   ├── types.ts
│   └── config.ts
├── modules/                 # Functional modules (already exists)
│   ├── analysis/
│   ├── validation/
│   ├── optimization/
│   ├── adaptation/
│   ├── storage/
│   ├── management/         # Model management (needs files moved here)
│   ├── utilities/
│   └── github/             # GitHub service (needs github-service.ts moved here)
├── services/                # High-level services (already exists)
│   ├── model-creation-service.ts
│   └── model-status-service.ts
├── hooks/                   # React hooks (already exists)
└── utils/                   # Shared utilities (already exists)
```

## ✅ Success Criteria

1. All files organized into proper modules
2. No duplicate functionality
3. All imports working
4. Tests passing
5. Clean root-level lib directory
