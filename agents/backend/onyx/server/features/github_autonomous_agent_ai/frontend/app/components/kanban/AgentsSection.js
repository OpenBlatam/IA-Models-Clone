'use client';
"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AgentsSection = AgentsSection;
exports.AgentsSection = AgentsSection;
var jsx_runtime_1 = require("react/jsx-runtime");
var react_1 = require("react");
var framer_motion_1 = require("framer-motion");
var link_1 = require("next/link");
var sonner_1 = require("sonner");
var cn_1 = require("../../utils/cn");
// Helper function to calculate success rate
function calculateSuccessRate(successful, total) {
    if (total === 0)
        return 0;
    return (successful * 100) / total;
}
function AgentsSection(_a) {
    var _this = this;
    var className = _a.className;
    var _b = (0, react_1.useState)([]), agents = _b[0], setAgents = _b[1];
    var _c = (0, react_1.useState)(true), isLoading = _c[0], setIsLoading = _c[1];
    var _d = (0, react_1.useState)(false), isExpanded = _d[0], setIsExpanded = _d[1];
    var _e = (0, react_1.useState)(''), searchQuery = _e[0], setSearchQuery = _e[1];
    var _f = (0, react_1.useState)('all'), filterStatus = _f[0], setFilterStatus = _f[1];
    var _g = (0, react_1.useState)('name'), sortBy = _g[0], setSortBy = _g[1];
    var _h = (0, react_1.useState)('asc'), sortOrder = _h[0], setSortOrder = _h[1];
    var _j = (0, react_1.useState)('detailed'), viewMode = _j[0], setViewMode = _j[1];
    var _k = (0, react_1.useState)(false), showCharts = _k[0], setShowCharts = _k[1];
    var _l = (0, react_1.useState)(null), selectedAgent = _l[0], setSelectedAgent = _l[1];
    var _m = (0, react_1.useState)(new Set()), selectedAgents = _m[0], setSelectedAgents = _m[1];
    var _o = (0, react_1.useState)([]), recentActivity = _o[0], setRecentActivity = _o[1];
    var _p = (0, react_1.useState)(false), showBulkActions = _p[0], setShowBulkActions = _p[1];
    var _q = (0, react_1.useState)('all'), filterTaskType = _q[0], setFilterTaskType = _q[1];
    var _r = (0, react_1.useState)('all'), filterCredits = _r[0], setFilterCredits = _r[1];
    var _s = (0, react_1.useState)(new Set()), comparingAgents = _s[0], setComparingAgents = _s[1];
    var _t = (0, react_1.useState)(false), showComparison = _t[0], setShowComparison = _t[1];
    var _u = (0, react_1.useState)(false), showAlerts = _u[0], setShowAlerts = _u[1];
    var _v = (0, react_1.useState)([]), savedFilters = _v[0], setSavedFilters = _v[1];
    var _w = (0, react_1.useState)(false), showExecutionCalendar = _w[0], setShowExecutionCalendar = _w[1];
    var _x = (0, react_1.useState)(false), showTrends = _x[0], setShowTrends = _x[1];
    var _y = (0, react_1.useState)('7d'), selectedTimeRange = _y[0], setSelectedTimeRange = _y[1];
    var fetchAgents = function () { return __awaiter(_this, void 0, void 0, function () {
        var response, data, error_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 4, 5, 6]);
                    return [4 /*yield*/, fetch('/api/continuous-agent')];
                case 1:
                    response = _a.sent();
                    if (!response.ok) return [3 /*break*/, 3];
                    return [4 /*yield*/, response.json()];
                case 2:
                    data = _a.sent();
                    setAgents(Array.isArray(data) ? data : []);
                    _a.label = 3;
                case 3: return [3 /*break*/, 6];
                case 4:
                    error_1 = _a.sent();
                    console.error('Error fetching agents:', error_1);
                    return [3 /*break*/, 6];
                case 5:
                    setIsLoading(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    (0, react_1.useEffect)(function () {
        fetchAgents();
        // Refresh every 10 seconds
        var interval = setInterval(fetchAgents, 10000);
        return function () { return clearInterval(interval); };
    }, []);
    // Detectar cambios en agentes para actividad reciente
    (0, react_1.useEffect)(function () {
        agents.forEach(function (agent) {
            // Esto se puede mejorar con un sistema de tracking más sofisticado
            // Por ahora, solo trackeamos cuando se detectan ejecuciones recientes
            if (agent.stats.lastExecutionAt) {
                var lastExec_1 = new Date(agent.stats.lastExecutionAt);
                var now = new Date();
                var diffMinutes = (now.getTime() - lastExec_1.getTime()) / 1000 / 60;
                if (diffMinutes < 5) {
                    setRecentActivity(function (prev) {
                        var exists = prev.some(function (a) {
                            return a.agentId === agent.id &&
                                a.action === 'executed' &&
                                Math.abs(a.timestamp.getTime() - lastExec_1.getTime()) < 60000;
                        });
                        if (!exists) {
                            return __spreadArray([{
                                    agentId: agent.id,
                                    agentName: agent.name,
                                    action: 'executed',
                                    timestamp: lastExec_1,
                                }], prev, true).slice(0, 10);
                        }
                        return prev;
                    });
                }
            }
        });
    }, [agents]);
    // Estadísticas agregadas
    var totalStats = (0, react_1.useMemo)(function () {
        return agents.reduce(function (acc, agent) { return ({
            totalExecutions: acc.totalExecutions + agent.stats.totalExecutions,
            successfulExecutions: acc.successfulExecutions + agent.stats.successfulExecutions,
            failedExecutions: acc.failedExecutions + agent.stats.failedExecutions,
            creditsUsed: acc.creditsUsed + agent.stats.creditsUsed,
            totalAgents: acc.totalAgents + 1,
        }); }, {
            totalExecutions: 0,
            successfulExecutions: 0,
            failedExecutions: 0,
            creditsUsed: 0,
            totalAgents: 0,
        });
    }, [agents]);
    // Filtrar y ordenar agentes
    var filteredAndSortedAgents = (0, react_1.useMemo)(function () {
        var filtered = agents;
        // Filtro por estado
        if (filterStatus === 'active') {
            filtered = filtered.filter(function (a) { return a.isActive; });
        }
        else if (filterStatus === 'inactive') {
            filtered = filtered.filter(function (a) { return !a.isActive; });
        }
        // Filtro por búsqueda
        if (searchQuery.trim()) {
            var query_1 = searchQuery.toLowerCase();
            filtered = filtered.filter(function (agent) {
                return agent.name.toLowerCase().includes(query_1) ||
                    agent.description.toLowerCase().includes(query_1) ||
                    agent.config.taskType.toLowerCase().includes(query_1);
            });
        }
        // Filtro por tipo de tarea
        if (filterTaskType !== 'all') {
            filtered = filtered.filter(function (agent) { return agent.config.taskType === filterTaskType; });
        }
        // Filtro por créditos
        if (filterCredits === 'with') {
            filtered = filtered.filter(function (agent) { return agent.stats.creditsUsed > 0; });
        }
        else if (filterCredits === 'without') {
            filtered = filtered.filter(function (agent) { return agent.stats.creditsUsed === 0; });
        }
        // Ordenamiento
        var sorted = __spreadArray([], filtered, true).sort(function (a, b) {
            var comparison = 0;
            switch (sortBy) {
                case 'name':
                    comparison = a.name.localeCompare(b.name);
                    break;
                case 'status':
                    comparison = Number(b.isActive) - Number(a.isActive);
                    break;
                case 'executions':
                    comparison = b.stats.totalExecutions - a.stats.totalExecutions;
                    break;
                case 'success':
                    var aRate = a.stats.totalExecutions > 0
                        ? a.stats.successfulExecutions / a.stats.totalExecutions
                        : 0;
                    var bRate = b.stats.totalExecutions > 0
                        ? b.stats.successfulExecutions / b.stats.totalExecutions
                        : 0;
                    comparison = bRate - aRate;
                    break;
                case 'created':
                    comparison = new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
                    break;
            }
            return sortOrder === 'asc' ? comparison : -comparison;
        });
        return sorted;
    }, [agents, filterStatus, searchQuery, sortBy, sortOrder, filterTaskType, filterCredits]);
    var activeAgents = filteredAndSortedAgents.filter(function (a) { return a.isActive; });
    var inactiveAgents = filteredAndSortedAgents.filter(function (a) { return !a.isActive; });
    // Tipos de tarea únicos para filtro
    var taskTypes = (0, react_1.useMemo)(function () {
        var types = new Set(agents.map(function (a) { return a.config.taskType; }));
        return Array.from(types).sort();
    }, [agents]);
    // Acciones en lote
    var handleBulkToggle = function (activate) { return __awaiter(_this, void 0, void 0, function () {
        var promises, results, successful, failed, error_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (selectedAgents.size === 0) {
                        sonner_1.toast.error('No hay agentes seleccionados', {
                            description: 'Selecciona al menos un agente para realizar la acción',
                        });
                        return [2 /*return*/];
                    }
                    promises = Array.from(selectedAgents).map(function (agentId) {
                        return fetch("/api/continuous-agent/".concat(agentId), {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ isActive: activate }),
                        });
                    });
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, Promise.allSettled(promises)];
                case 2:
                    results = _a.sent();
                    successful = results.filter(function (r) { return r.status === 'fulfilled'; }).length;
                    failed = results.length - successful;
                    if (successful > 0) {
                        sonner_1.toast.success("".concat(successful, " agente(s) ").concat(activate ? 'activado(s)' : 'pausado(s)'), {
                            description: failed > 0 ? "".concat(failed, " fallaron") : undefined,
                        });
                        setSelectedAgents(new Set());
                        fetchAgents();
                    }
                    else {
                        sonner_1.toast.error('Error al actualizar agentes', {
                            description: 'No se pudo cambiar el estado de los agentes',
                        });
                    }
                    return [3 /*break*/, 4];
                case 3:
                    error_2 = _a.sent();
                    sonner_1.toast.error('Error al actualizar agentes', {
                        description: 'Ocurrió un error al procesar la acción',
                    });
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleSelectAll = function () {
        if (selectedAgents.size === filteredAndSortedAgents.length) {
            setSelectedAgents(new Set());
        }
        else {
            setSelectedAgents(new Set(filteredAndSortedAgents.map(function (a) { return a.id; })));
        }
    };
    var handleSelectAgent = function (agentId) {
        setSelectedAgents(function (prev) {
            var newSet = new Set(prev);
            if (newSet.has(agentId)) {
                newSet.delete(agentId);
            }
            else {
                newSet.add(agentId);
            }
            return newSet;
        });
    };
    var handleCompareAgent = function (agentId) {
        setComparingAgents(function (prev) {
            var newSet = new Set(prev);
            if (newSet.has(agentId)) {
                newSet.delete(agentId);
            }
            else {
                if (newSet.size < 3) {
                    newSet.add(agentId);
                }
                else {
                    sonner_1.toast.error('Máximo 3 agentes para comparar', {
                        description: 'Deselecciona un agente antes de agregar otro',
                    });
                }
            }
            return newSet;
        });
    };
    var handleSaveFilter = function () {
        var name = prompt('Nombre para este filtro:');
        if (name) {
            setSavedFilters(function (prev) { return __spreadArray(__spreadArray([], prev, true), [{
                    name: name,
                    filters: {
                        searchQuery: searchQuery,
                        filterStatus: filterStatus,
                        filterTaskType: filterTaskType,
                        filterCredits: filterCredits,
                    },
                }], false); });
            sonner_1.toast.success('Filtro guardado', {
                description: "Filtro \"".concat(name, "\" guardado correctamente"),
            });
        }
    };
    var handleLoadFilter = function (filter) {
        setSearchQuery(filter.filters.searchQuery);
        setFilterStatus(filter.filters.filterStatus);
        setFilterTaskType(filter.filters.filterTaskType);
        setFilterCredits(filter.filters.filterCredits);
        sonner_1.toast.success('Filtro aplicado', {
            description: "Filtro \"".concat(filter.name, "\" aplicado"),
        });
    };
    // Distribución por tipo de tarea
    var taskTypeDistribution = (0, react_1.useMemo)(function () {
        var distribution = {};
        agents.forEach(function (agent) {
            var type = agent.config.taskType;
            distribution[type] = (distribution[type] || 0) + 1;
        });
        return distribution;
    }, [agents]);
    // Agentes con mejor rendimiento
    var topPerformers = (0, react_1.useMemo)(function () {
        return __spreadArray([], agents, true).filter(function (a) { return a.stats.totalExecutions > 0; })
            .map(function (a) { return ({
            agent: a,
            successRate: (a.stats.successfulExecutions / a.stats.totalExecutions) * 100,
        }); })
            .sort(function (a, b) { return b.successRate - a.successRate; })
            .slice(0, 3);
    }, [agents]);
    // Agentes más activos
    var mostActive = (0, react_1.useMemo)(function () {
        return __spreadArray([], agents, true).filter(function (a) { return a.isActive; })
            .sort(function (a, b) { return b.stats.totalExecutions - a.stats.totalExecutions; })
            .slice(0, 3);
    }, [agents]);
    // Alertas y notificaciones
    var alerts = (0, react_1.useMemo)(function () {
        var alertsList = [];
        agents.forEach(function (agent) {
            // Agentes inactivos por mucho tiempo
            if (!agent.isActive && agent.stats.lastExecutionAt) {
                var lastExec = new Date(agent.stats.lastExecutionAt);
                var daysSince = (Date.now() - lastExec.getTime()) / (1000 * 60 * 60 * 24);
                if (daysSince > 7) {
                    alertsList.push({
                        type: 'info',
                        message: "Inactivo por ".concat(Math.floor(daysSince), " d\u00EDas"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
            }
            // Alta tasa de fallos
            if (agent.stats.totalExecutions > 10) {
                var failureRate = (agent.stats.failedExecutions / agent.stats.totalExecutions) * 100;
                if (failureRate > 50) {
                    alertsList.push({
                        type: 'error',
                        message: "Tasa de fallos alta: ".concat(failureRate.toFixed(1), "%"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
                else if (failureRate > 30) {
                    alertsList.push({
                        type: 'warning',
                        message: "Tasa de fallos moderada: ".concat(failureRate.toFixed(1), "%"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
            }
            // Sin créditos
            if (agent.stripeCreditsRemaining !== null && agent.stripeCreditsRemaining < 1) {
                alertsList.push({
                    type: 'warning',
                    message: 'Sin créditos disponibles',
                    agentId: agent.id,
                    agentName: agent.name,
                });
            }
            // Sin ejecuciones recientes en agentes activos
            if (agent.isActive && agent.stats.lastExecutionAt) {
                var lastExec = new Date(agent.stats.lastExecutionAt);
                var hoursSince = (Date.now() - lastExec.getTime()) / (1000 * 60 * 60);
                var expectedExecutions = Math.floor(hoursSince / (agent.config.frequency / 3600));
                if (expectedExecutions > 3 && agent.stats.totalExecutions > 0) {
                    alertsList.push({
                        type: 'warning',
                        message: "Sin ejecuciones en ".concat(Math.floor(hoursSince), " horas"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
            }
        });
        return alertsList;
    }, [agents]);
    // Agentes para comparación
    var agentsToCompare = (0, react_1.useMemo)(function () {
        return Array.from(comparingAgents)
            .map(function (id) { return agents.find(function (a) { return a.id === id; }); })
            .filter(function (a) { return a !== undefined; });
    }, [comparingAgents, agents]);
    // Análisis de prompts
    var promptAnalytics = (0, react_1.useMemo)(function () {
        var agentsWithGoals = agents.filter(function (a) { return a.config.goal && a.config.goal.trim().length > 0; });
        var totalGoalsLength = agentsWithGoals.reduce(function (sum, a) { var _a; return sum + (((_a = a.config.goal) === null || _a === void 0 ? void 0 : _a.length) || 0); }, 0);
        var averageGoalLength = agentsWithGoals.length > 0 ? totalGoalsLength / agentsWithGoals.length : 0;
        // Análisis de salud de prompts
        var healthyGoals = agentsWithGoals.filter(function (a) {
            var goal = a.config.goal || '';
            return goal.length >= 50 && goal.length <= 1000;
        });
        var healthScore = agentsWithGoals.length > 0
            ? Math.round((healthyGoals.length / agentsWithGoals.length) * 100)
            : 0;
        return {
            agentsWithGoals: agentsWithGoals.length,
            totalAgents: agents.length,
            averageGoalLength: Math.round(averageGoalLength),
            healthyGoals: healthyGoals.length,
            healthScore: healthScore,
        };
    }, [agents]);
    // Análisis de rendimiento por tipo de tarea
    var taskTypePerformance = (0, react_1.useMemo)(function () {
        var performance = {};
        agents.forEach(function (agent) {
            var type = agent.config.taskType;
            if (!performance[type]) {
                performance[type] = {
                    count: 0,
                    totalExecutions: 0,
                    successfulExecutions: 0,
                    averageSuccessRate: 0,
                };
            }
            performance[type].count++;
            performance[type].totalExecutions += agent.stats.totalExecutions;
            performance[type].successfulExecutions += agent.stats.successfulExecutions;
        });
        // Calcular tasa promedio de éxito por tipo
        Object.keys(performance).forEach(function (type) {
            var perf = performance[type];
            perf.averageSuccessRate = perf.totalExecutions > 0
                ? (perf.successfulExecutions / perf.totalExecutions) * 100
                : 0;
        });
        return performance;
    }, [agents]);
    // Próximas ejecuciones para calendario
    var upcomingExecutions = (0, react_1.useMemo)(function () {
        return agents
            .filter(function (a) { return a.isActive && a.stats.nextExecutionAt; })
            .map(function (agent) { return ({
            agent: agent,
            nextExecution: new Date(agent.stats.nextExecutionAt),
        }); })
            .sort(function (a, b) { return a.nextExecution.getTime() - b.nextExecution.getTime(); })
            .slice(0, 10);
    }, [agents]);
    // Análisis de tendencias temporales
    var executionTrends = (0, react_1.useMemo)(function () {
        var now = new Date();
        var ranges = {
            '24h': 24 * 60 * 60 * 1000,
            '7d': 7 * 24 * 60 * 60 * 1000,
            '30d': 30 * 24 * 60 * 60 * 1000,
        };
        var timeRange = ranges[selectedTimeRange] || Infinity;
        var cutoffTime = selectedTimeRange === 'all' ? 0 : now.getTime() - timeRange;
        var recentExecutions = agents
            .filter(function (a) { return a.stats.lastExecutionAt && new Date(a.stats.lastExecutionAt).getTime() >= cutoffTime; })
            .length;
        var activeAgentsCount = agents.filter(function (a) { return a.isActive; }).length;
        var avgExecutionsPerAgent = agents.length > 0
            ? agents.reduce(function (sum, a) { return sum + a.stats.totalExecutions; }, 0) / agents.length
            : 0;
        // Calcular tendencia (comparar con período anterior si es posible)
        var trend = 'stable';
        if (selectedTimeRange !== 'all') {
            var previousRange = selectedTimeRange === '24h' ? '7d' : selectedTimeRange === '7d' ? '30d' : 'all';
            var previousCutoff_1 = previousRange === 'all' ? 0 : now.getTime() - (ranges[previousRange] || Infinity);
            var previousExecutions = agents
                .filter(function (a) {
                if (!a.stats.lastExecutionAt)
                    return false;
                var execTime = new Date(a.stats.lastExecutionAt).getTime();
                return execTime >= previousCutoff_1 && execTime < cutoffTime;
            })
                .length;
            if (recentExecutions > previousExecutions * 1.1)
                trend = 'up';
            else if (recentExecutions < previousExecutions * 0.9)
                trend = 'down';
        }
        return {
            recentExecutions: recentExecutions,
            activeAgentsCount: activeAgentsCount,
            avgExecutionsPerAgent: Math.round(avgExecutionsPerAgent * 10) / 10,
            trend: trend,
        };
    }, [agents, selectedTimeRange]);
    // Agentes con mejor rendimiento reciente
    var recentTopPerformers = (0, react_1.useMemo)(function () {
        var now = new Date();
        var sevenDaysAgo = now.getTime() - (7 * 24 * 60 * 60 * 1000);
        return __spreadArray([], agents, true).filter(function (a) {
            if (!a.stats.lastExecutionAt)
                return false;
            return new Date(a.stats.lastExecutionAt).getTime() >= sevenDaysAgo;
        })
            .map(function (a) { return ({
            agent: a,
            recentSuccessRate: a.stats.totalExecutions > 0
                ? (a.stats.successfulExecutions / a.stats.totalExecutions) * 100
                : 0,
        }); })
            .sort(function (a, b) { return b.recentSuccessRate - a.recentSuccessRate; })
            .slice(0, 5);
    }, [agents]);
    // Función para exportar agentes
    var handleExport = function () {
        var dataStr = JSON.stringify(filteredAndSortedAgents, null, 2);
        var dataBlob = new Blob([dataStr], { type: 'application/json' });
        var url = URL.createObjectURL(dataBlob);
        var link = document.createElement('a');
        link.href = url;
        link.download = "agentes-".concat(new Date().toISOString().split('T')[0], ".json");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        sonner_1.toast.success('Agentes exportados', {
            description: "".concat(filteredAndSortedAgents.length, " agente(s) exportado(s) correctamente"),
        });
    };
    // Función para exportar CSV
    var handleExportCSV = function () {
        var headers = ['Nombre', 'Estado', 'Tipo', 'Ejecuciones', 'Exitosas', 'Fallidas', 'Tasa Éxito', 'Créditos Usados', 'Creado'];
        var rows = filteredAndSortedAgents.map(function (agent) {
            var successRate = agent.stats.totalExecutions > 0
                ? ((agent.stats.successfulExecutions / agent.stats.totalExecutions) * 100).toFixed(2)
                : '0.00';
            return [
                agent.name,
                agent.isActive ? 'Activo' : 'Inactivo',
                agent.config.taskType,
                agent.stats.totalExecutions.toString(),
                agent.stats.successfulExecutions.toString(),
                agent.stats.failedExecutions.toString(),
                "".concat(successRate, "%"),
                agent.stats.creditsUsed.toFixed(2),
                new Date(agent.createdAt).toLocaleDateString('es-ES'),
            ];
        });
        var csvContent = __spreadArray([
            headers.join(',')
        ], rows.map(function (row) { return row.map(function (cell) { return "\"".concat(cell, "\""); }).join(','); }), true).join('\n');
        var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        var url = URL.createObjectURL(blob);
        var link = document.createElement('a');
        link.href = url;
        link.download = "agentes-".concat(new Date().toISOString().split('T')[0], ".csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        sonner_1.toast.success('Agentes exportados a CSV', {
            description: "".concat(filteredAndSortedAgents.length, " agente(s) exportado(s) correctamente"),
        });
    };
    if (isLoading) {
        return ((0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("bg-white border border-gray-200 rounded-lg p-4", className), children: (0, jsx_runtime_1.jsxs)("div", { className: "animate-pulse", children: [(0, jsx_runtime_1.jsx)("div", { className: "h-4 bg-gray-200 rounded w-1/4 mb-3" }), (0, jsx_runtime_1.jsx)("div", { className: "h-3 bg-gray-200 rounded w-1/2" })] }) }));
    }
    if (agents.length === 0) {
        return null;
    }
    return ((0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("bg-white border border-gray-200 rounded-lg shadow-sm", className), children: [(0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 flex items-center justify-between border-b border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-3 flex-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-5 h-5 text-gray-600", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" }) }), (0, jsx_runtime_1.jsx)("h3", { className: "font-semibold text-gray-900", children: "Agentes Continuos" })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [activeAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full", children: [activeAgents.length, " activo", activeAgents.length !== 1 ? 's' : ''] })), (0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-gray-100 text-gray-700 text-xs font-medium rounded-full", children: [filteredAndSortedAgents.length, " / ", agents.length] }), selectedAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full", children: [selectedAgents.size, " seleccionado", selectedAgents.size !== 1 ? 's' : ''] })), comparingAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full", children: [comparingAgents.size, " comparando"] })), alerts.length > 0 && ((0, jsx_runtime_1.jsxs)("button", { onClick: function () { return setShowAlerts(!showAlerts); }, className: (0, cn_1.cn)("px-2 py-0.5 text-xs font-medium rounded-full transition-colors relative", showAlerts
                                            ? "bg-yellow-100 text-yellow-700"
                                            : "bg-red-100 text-red-700 hover:bg-red-200"), children: [alerts.length, " alerta", alerts.length !== 1 ? 's' : '', (0, jsx_runtime_1.jsx)("span", { className: "absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" })] }))] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [isExpanded && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return setShowCharts(!showCharts); }, className: (0, cn_1.cn)("px-2 py-1.5 text-xs rounded-lg transition-colors flex items-center gap-1", showCharts
                                            ? "bg-blue-100 text-blue-700 hover:bg-blue-200"
                                            : "text-gray-600 hover:text-gray-700 hover:bg-gray-100"), title: "Mostrar gr\u00E1ficos", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }) }), (0, jsx_runtime_1.jsxs)("div", { className: "relative group", children: [(0, jsx_runtime_1.jsx)("button", { onClick: handleExport, className: "px-2 py-1.5 text-xs text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors flex items-center gap-1", title: "Exportar agentes", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" }) }) }), (0, jsx_runtime_1.jsxs)("div", { className: "absolute right-0 top-full mt-1 w-40 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50", children: [(0, jsx_runtime_1.jsx)("button", { onClick: handleExport, className: "w-full text-left px-3 py-2 text-xs hover:bg-gray-50 rounded-t-lg", children: "Exportar JSON" }), (0, jsx_runtime_1.jsx)("button", { onClick: handleExportCSV, className: "w-full text-left px-3 py-2 text-xs hover:bg-gray-50 rounded-b-lg", children: "Exportar CSV" })] })] }), (0, jsx_runtime_1.jsx)("div", { className: "relative group", children: (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                var modes = ['detailed', 'compact', 'table'];
                                                var currentIndex = modes.indexOf(viewMode);
                                                setViewMode(modes[(currentIndex + 1) % modes.length]);
                                            }, className: "px-2 py-1.5 text-xs text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors", title: "Vista: ".concat(viewMode === 'compact' ? 'Compacta' : viewMode === 'table' ? 'Tabla' : 'Detallada'), children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: viewMode === 'compact' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" })) : viewMode === 'table' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" })) : ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M4 6h16M4 12h16M4 18h16" })) }) }) }), selectedAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return setShowBulkActions(!showBulkActions); }, className: "px-2 py-1.5 text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 rounded-lg transition-colors", title: "Acciones en lote", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" }) }) }), showBulkActions && ((0, jsx_runtime_1.jsxs)("div", { className: "absolute right-0 top-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 flex gap-1 p-1", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(true); }, className: "px-2 py-1 text-xs bg-green-100 text-green-700 hover:bg-green-200 rounded transition-colors", children: "Activar todos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(false); }, className: "px-2 py-1 text-xs bg-red-100 text-red-700 hover:bg-red-200 rounded transition-colors", children: "Pausar todos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setSelectedAgents(new Set()); }, className: "px-2 py-1 text-xs bg-gray-100 text-gray-700 hover:bg-gray-200 rounded transition-colors", children: "Limpiar" })] }))] }))] })), (0, jsx_runtime_1.jsxs)(link_1.default, { href: "/continuous-agent", className: "px-3 py-1.5 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors flex items-center gap-1", title: "Ver todos los agentes", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" }) }), "Ver todos"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setIsExpanded(!isExpanded); }, className: "p-1.5 hover:bg-gray-100 rounded-lg transition-colors", title: isExpanded ? "Contraer" : "Expandir", children: (0, jsx_runtime_1.jsx)("svg", { className: (0, cn_1.cn)("w-5 h-5 text-gray-500 transition-transform", isExpanded && "transform rotate-180"), fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M19 9l-7 7-7-7" }) }) })] })] }), isExpanded && totalStats.totalAgents > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 bg-gray-50 border-b border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-3 text-xs", children: [(0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Total Ejecuciones" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900 text-lg", children: totalStats.totalExecutions.toLocaleString() })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Exitosas" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-green-600 text-lg", children: totalStats.successfulExecutions.toLocaleString() })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Fallidas" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-red-600 text-lg", children: totalStats.failedExecutions.toLocaleString() })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Cr\u00E9ditos Usados" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-blue-600 text-lg", children: totalStats.creditsUsed.toFixed(2) })] })] }), totalStats.totalExecutions > 0 && (function () {
                        var globalSuccessRate = calculateSuccessRate(totalStats.successfulExecutions, totalStats.totalExecutions);
                        return ((0, jsx_runtime_1.jsxs)("div", { className: "mt-3 p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs mb-1", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-500 font-medium", children: "Tasa de \u00E9xito global" }), (0, jsx_runtime_1.jsxs)("span", { className: "font-bold text-gray-900 text-sm", children: [globalSuccessRate.toFixed(1), "%"] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-2", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-2 rounded-full transition-all", globalSuccessRate >= 80 ? "bg-green-500" :
                                            globalSuccessRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: {
                                            width: "".concat(globalSuccessRate, "%")
                                        } }) })] }));
                    })(), (0, jsx_runtime_1.jsxs)("div", { className: "mt-2 grid grid-cols-2 md:grid-cols-3 gap-2 text-xs", children: [(0, jsx_runtime_1.jsxs)("div", { className: "p-1.5 bg-white rounded border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Agentes Totales" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900", children: totalStats.totalAgents })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-1.5 bg-white rounded border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Activos" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-green-600", children: activeAgents.length })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-1.5 bg-white rounded border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Inactivos" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-600", children: inactiveAgents.length })] })] })] })), isExpanded && showCharts && totalStats.totalAgents > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 bg-white border-b border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [(0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700", children: "Distribuci\u00F3n por Tipo" }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-1.5", children: Object.entries(taskTypeDistribution).map(function (_a) {
                                            var type = _a[0], count = _a[1];
                                            var percentage = (count / agents.length) * 100;
                                            return ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-600 capitalize", children: type.replace(new RegExp('_', 'g'), ' ') }), (0, jsx_runtime_1.jsxs)("span", { className: "font-medium text-gray-900", children: [count, " (", percentage.toFixed(0), "%)"] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-2", children: (0, jsx_runtime_1.jsx)("div", { className: "bg-blue-500 h-2 rounded-full transition-all", style: { width: "".concat(percentage, "%") } }) })] }, type));
                                        }) })] }), (0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700", children: "Mejor Rendimiento" }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-2", children: topPerformers.length > 0 ? (topPerformers.map(function (_a, index) {
                                            var agent = _a.agent, successRate = _a.successRate;
                                            return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 p-2 bg-gray-50 rounded-lg", children: [(0, jsx_runtime_1.jsx)("div", { className: "flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center text-xs font-bold text-white", children: index + 1 }), (0, jsx_runtime_1.jsxs)("div", { className: "flex-1 min-w-0", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-xs font-medium text-gray-900 truncate", children: agent.name }), (0, jsx_runtime_1.jsxs)("div", { className: "text-xs text-gray-500", children: [successRate.toFixed(1), "% \u00E9xito \u2022 ", agent.stats.totalExecutions, " ejec."] })] }), (0, jsx_runtime_1.jsx)("div", { className: "flex-shrink-0", children: (0, jsx_runtime_1.jsxs)("div", { className: "w-12 h-12 relative", children: [(0, jsx_runtime_1.jsxs)("svg", { className: "transform -rotate-90", viewBox: "0 0 36 36", children: [(0, jsx_runtime_1.jsx)("circle", { cx: "18", cy: "18", r: "16", fill: "none", stroke: "#e5e7eb", strokeWidth: "3" }), (0, jsx_runtime_1.jsx)("circle", { cx: "18", cy: "18", r: "16", fill: "none", stroke: successRate >= 80 ? "#10b981" : successRate >= 50 ? "#f59e0b" : "#ef4444", strokeWidth: "3", strokeDasharray: "".concat(successRate, ", 100"), strokeLinecap: "round" })] }), (0, jsx_runtime_1.jsxs)("div", { className: "absolute inset-0 flex items-center justify-center text-xs font-semibold text-gray-900", children: [successRate.toFixed(0), "%"] })] }) })] }, agent.id));
                                        })) : ((0, jsx_runtime_1.jsx)("p", { className: "text-xs text-gray-500 text-center py-2", children: "No hay datos suficientes" })) })] })] }), mostActive.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-4 pt-4 border-t border-gray-200", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700 mb-2", children: "M\u00E1s Activos" }), (0, jsx_runtime_1.jsx)("div", { className: "flex gap-2", children: mostActive.map(function (agent, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: "flex-1 p-2 bg-gradient-to-br from-green-50 to-blue-50 rounded-lg border border-green-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-xs font-medium text-gray-900 truncate", children: agent.name }), (0, jsx_runtime_1.jsxs)("div", { className: "text-xs text-gray-600 mt-1", children: [agent.stats.totalExecutions, " ejecuciones"] }), (0, jsx_runtime_1.jsxs)("div", { className: "mt-1 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("div", { className: "flex-1 bg-gray-200 rounded-full h-1", children: (function () {
                                                        var _a;
                                                        var maxExecutions = ((_a = mostActive[0]) === null || _a === void 0 ? void 0 : _a.stats.totalExecutions) || 1;
                                                        var executionPercentage = Math.min((agent.stats.totalExecutions / maxExecutions) * 100, 100);
                                                        return ((0, jsx_runtime_1.jsx)("div", { className: "bg-green-500 h-1 rounded-full", style: {
                                                                width: "".concat(executionPercentage, "%")
                                                            } }));
                                                    })() }), (0, jsx_runtime_1.jsxs)("span", { className: "text-xs text-gray-500", children: ["#", index + 1] })] })] }, agent.id)); }) })] })), promptAnalytics.agentsWithGoals > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-4 pt-4 border-t border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-2", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700", children: "Estad\u00EDsticas de Prompts" }), (0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("text-xs font-bold", promptAnalytics.healthScore >= 80 ? "text-green-600" :
                                            promptAnalytics.healthScore >= 60 ? "text-yellow-600" : "text-red-600"), children: ["Salud: ", promptAnalytics.healthScore, "%"] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "grid grid-cols-2 gap-2 text-xs", children: [(0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-gray-50 rounded", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900", children: promptAnalytics.agentsWithGoals }), (0, jsx_runtime_1.jsxs)("div", { className: "text-gray-500", children: ["Con objetivos (", Math.round((promptAnalytics.agentsWithGoals / promptAnalytics.totalAgents) * 100), "%)"] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-gray-50 rounded", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-blue-600", children: promptAnalytics.healthyGoals }), (0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Objetivos saludables" })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-gray-50 rounded col-span-2", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900", children: promptAnalytics.averageGoalLength.toLocaleString() }), (0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Promedio de caracteres" })] })] })] })), Object.keys(taskTypePerformance).length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-4 pt-4 border-t border-gray-200", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700 mb-2", children: "Rendimiento por Tipo" }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-2", children: Object.entries(taskTypePerformance)
                                    .sort(function (a, b) { return b[1].averageSuccessRate - a[1].averageSuccessRate; })
                                    .map(function (_a) {
                                    var type = _a[0], perf = _a[1];
                                    return ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-600 capitalize", children: type.replace(new RegExp('_', 'g'), ' ') }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsxs)("span", { className: "text-gray-500", children: [perf.count, " agente", perf.count !== 1 ? 's' : ''] }), (0, jsx_runtime_1.jsxs)("span", { className: (0, cn_1.cn)("font-medium", perf.averageSuccessRate >= 80 ? "text-green-600" :
                                                                    perf.averageSuccessRate >= 50 ? "text-yellow-600" : "text-red-600"), children: [perf.averageSuccessRate.toFixed(1), "% \u00E9xito"] })] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-1.5", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-1.5 rounded-full transition-all", perf.averageSuccessRate >= 80 ? "bg-green-500" :
                                                        perf.averageSuccessRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: { width: "".concat(Math.min(perf.averageSuccessRate, 100), "%") } }) }), (0, jsx_runtime_1.jsxs)("div", { className: "text-xs text-gray-400", children: [perf.totalExecutions, " ejecuciones totales"] })] }, type));
                                }) })] }))] })), isExpanded && showAlerts && alerts.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 bg-red-50 border-b border-red-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-2", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-red-700 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" }) }), "Alertas (", alerts.length, ")"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setShowAlerts(false); }, className: "text-xs text-red-600 hover:text-red-700", children: "Cerrar" })] }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-1.5 max-h-32 overflow-y-auto", children: alerts.map(function (alert, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("p-2 rounded-lg text-xs flex items-start gap-2", alert.type === 'error' ? "bg-red-100 text-red-800" :
                                alert.type === 'warning' ? "bg-yellow-100 text-yellow-800" :
                                    "bg-blue-100 text-blue-800"), children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4 flex-shrink-0 mt-0.5", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: alert.type === 'error' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" })) : alert.type === 'warning' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" })) : ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" })) }), (0, jsx_runtime_1.jsxs)("div", { className: "flex-1", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-medium", children: alert.agentName || 'Sistema' }), (0, jsx_runtime_1.jsx)("div", { children: alert.message })] })] }, index)); }) })] })), isExpanded && showComparison && agentsToCompare.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 bg-purple-50 border-b border-purple-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-3", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-purple-700 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }), "Comparaci\u00F3n (", agentsToCompare.length, ")"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                    setComparingAgents(new Set());
                                    setShowComparison(false);
                                }, className: "text-xs text-purple-600 hover:text-purple-700", children: "Cerrar" })] }), (0, jsx_runtime_1.jsx)("div", { className: "overflow-x-auto", children: (0, jsx_runtime_1.jsxs)("table", { className: "w-full text-xs", children: [(0, jsx_runtime_1.jsx)("thead", { children: (0, jsx_runtime_1.jsxs)("tr", { className: "border-b border-purple-200", children: [(0, jsx_runtime_1.jsx)("th", { className: "px-2 py-1 text-left text-purple-700", children: "M\u00E9trica" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("th", { className: "px-2 py-1 text-left text-purple-700", children: agent.name }, agent.id)); })] }) }), (0, jsx_runtime_1.jsxs)("tbody", { className: "divide-y divide-purple-100", children: [(0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Estado" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1", children: (0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("px-1.5 py-0.5 text-xs rounded-full", agent.isActive ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-700"), children: agent.isActive ? 'Activo' : 'Inactivo' }) }, agent.id)); })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Ejecuciones" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 text-gray-600", children: agent.stats.totalExecutions }, agent.id)); })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Tasa \u00C9xito" }), agentsToCompare.map(function (agent) {
                                                    var rate = calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions);
                                                    return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1", children: (0, jsx_runtime_1.jsx)("div", { className: "flex items-center gap-1", children: (0, jsx_runtime_1.jsxs)("span", { className: (0, cn_1.cn)("text-xs font-medium", rate >= 80 ? "text-green-600" : rate >= 50 ? "text-yellow-600" : "text-red-600"), children: [rate.toFixed(1), "%"] }) }) }, agent.id));
                                                })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Cr\u00E9ditos Usados" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 text-gray-600", children: agent.stats.creditsUsed.toFixed(2) }, agent.id)); })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Tiempo Promedio" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 text-gray-600", children: agent.stats.averageExecutionTime > 0
                                                        ? "".concat(agent.stats.averageExecutionTime.toFixed(2), "s")
                                                        : '-' }, agent.id)); })] })] })] }) })] })), isExpanded && recentActivity.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 bg-blue-50 border-b border-blue-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-2", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-gray-700 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 10V3L4 14h7v7l9-11h-7z" }) }), "Actividad Reciente"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setRecentActivity([]); }, className: "text-xs text-gray-500 hover:text-gray-700", children: "Limpiar" })] }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-1 max-h-24 overflow-y-auto", children: recentActivity.slice(0, 5).map(function (activity, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 text-xs", children: [(0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("w-1.5 h-1.5 rounded-full", activity.action === 'executed' ? "bg-green-500" :
                                        activity.action === 'activated' ? "bg-blue-500" : "bg-gray-400") }), (0, jsx_runtime_1.jsx)("span", { className: "text-gray-600 truncate", children: activity.agentName }), (0, jsx_runtime_1.jsx)("span", { className: "text-gray-400", children: activity.action === 'executed' ? 'ejecutó' :
                                        activity.action === 'activated' ? 'activado' : 'desactivado' }), (0, jsx_runtime_1.jsx)("span", { className: "text-gray-400 ml-auto", children: new Date(activity.timestamp).toLocaleTimeString('es-ES', {
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    }) })] }, index)); }) })] })), isExpanded && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 border-b border-gray-200 bg-gray-50 space-y-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex flex-col sm:flex-row gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex-1 relative", children: [(0, jsx_runtime_1.jsx)("input", { type: "text", placeholder: "Buscar agentes...", value: searchQuery, onChange: function (e) { return setSearchQuery(e.target.value); }, className: "w-full pl-8 pr-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" }), (0, jsx_runtime_1.jsx)("svg", { className: "absolute left-2.5 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" }) })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex gap-1", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return setFilterStatus('all'); }, className: (0, cn_1.cn)("px-3 py-1.5 text-xs font-medium rounded-lg transition-colors", filterStatus === 'all'
                                            ? "bg-blue-100 text-blue-700 border border-blue-300"
                                            : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"), children: "Todos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setFilterStatus('active'); }, className: (0, cn_1.cn)("px-3 py-1.5 text-xs font-medium rounded-lg transition-colors", filterStatus === 'active'
                                            ? "bg-green-100 text-green-700 border border-green-300"
                                            : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"), children: "Activos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setFilterStatus('inactive'); }, className: (0, cn_1.cn)("px-3 py-1.5 text-xs font-medium rounded-lg transition-colors", filterStatus === 'inactive'
                                            ? "bg-gray-100 text-gray-700 border border-gray-300"
                                            : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"), children: "Inactivos" })] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex flex-wrap items-center gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Ordenar por:" }), (0, jsx_runtime_1.jsxs)("select", { value: sortBy, onChange: function (e) { return setSortBy(e.target.value); }, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500", children: [(0, jsx_runtime_1.jsx)("option", { value: "name", children: "Nombre" }), (0, jsx_runtime_1.jsx)("option", { value: "status", children: "Estado" }), (0, jsx_runtime_1.jsx)("option", { value: "executions", children: "Ejecuciones" }), (0, jsx_runtime_1.jsx)("option", { value: "success", children: "Tasa \u00E9xito" }), (0, jsx_runtime_1.jsx)("option", { value: "created", children: "Fecha creaci\u00F3n" })] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); }, className: "p-1.5 border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors", title: sortOrder === 'asc' ? 'Ascendente' : 'Descendente', children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: sortOrder === 'asc' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M5 15l7-7 7 7" })) : ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M19 9l-7 7-7-7" })) }) })] }), taskTypes.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Tipo:" }), (0, jsx_runtime_1.jsxs)("select", { value: filterTaskType, onChange: function (e) { return setFilterTaskType(e.target.value); }, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500", children: [(0, jsx_runtime_1.jsx)("option", { value: "all", children: "Todos" }), taskTypes.map(function (type) { return ((0, jsx_runtime_1.jsx)("option", { value: type, children: type.replace(new RegExp('_', 'g'), ' ').replace(new RegExp('\\b\\w', 'g'), function (l) { return l.toUpperCase(); }) }, type)); })] })] })), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Cr\u00E9ditos:" }), (0, jsx_runtime_1.jsxs)("select", { value: filterCredits, onChange: function (e) { return setFilterCredits(e.target.value); }, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500", children: [(0, jsx_runtime_1.jsx)("option", { value: "all", children: "Todos" }), (0, jsx_runtime_1.jsx)("option", { value: "with", children: "Con cr\u00E9ditos usados" }), (0, jsx_runtime_1.jsx)("option", { value: "without", children: "Sin cr\u00E9ditos usados" })] })] }), savedFilters.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Filtros guardados:" }), (0, jsx_runtime_1.jsxs)("div", { className: "relative group", children: [(0, jsx_runtime_1.jsxs)("button", { className: "px-2 py-1 text-xs border border-gray-300 rounded-lg hover:bg-gray-50", children: [savedFilters.length, " guardado", savedFilters.length !== 1 ? 's' : ''] }), (0, jsx_runtime_1.jsxs)("div", { className: "absolute right-0 top-full mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50", children: [(0, jsx_runtime_1.jsx)("div", { className: "p-1 max-h-48 overflow-y-auto", children: savedFilters.map(function (filter, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between p-2 hover:bg-gray-50 rounded", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleLoadFilter(filter); }, className: "flex-1 text-left text-xs text-gray-700 hover:text-blue-600", children: filter.name }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                                        setSavedFilters(function (prev) { return prev.filter(function (_, i) { return i !== index; }); });
                                                                        sonner_1.toast.success('Filtro eliminado');
                                                                    }, className: "p-1 text-red-600 hover:text-red-800", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M6 18L18 6M6 6l12 12" }) }) })] }, index)); }) }), (0, jsx_runtime_1.jsx)("div", { className: "border-t border-gray-200 p-1", children: (0, jsx_runtime_1.jsx)("button", { onClick: handleSaveFilter, className: "w-full px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded text-center", children: "+ Guardar filtros actuales" }) })] })] })] })), savedFilters.length === 0 && ((0, jsx_runtime_1.jsx)("button", { onClick: handleSaveFilter, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg hover:bg-gray-50 text-gray-600", title: "Guardar filtros actuales", children: "Guardar filtros" }))] })] })), isExpanded && selectedAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 bg-blue-50 border-b border-blue-200 flex items-center justify-between", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("button", { onClick: handleSelectAll, className: "text-xs text-blue-600 hover:text-blue-700 font-medium", children: selectedAgents.size === filteredAndSortedAgents.length ? 'Deseleccionar todos' : 'Seleccionar todos' }), (0, jsx_runtime_1.jsxs)("span", { className: "text-xs text-gray-600", children: [selectedAgents.size, " de ", filteredAndSortedAgents.length, " seleccionado", selectedAgents.size !== 1 ? 's' : ''] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(true); }, className: "px-3 py-1 text-xs bg-green-100 text-green-700 hover:bg-green-200 rounded-lg transition-colors", children: "Activar seleccionados" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(false); }, className: "px-3 py-1 text-xs bg-red-100 text-red-700 hover:bg-red-200 rounded-lg transition-colors", children: "Pausar seleccionados" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setSelectedAgents(new Set()); }, className: "px-3 py-1 text-xs bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors", children: "Limpiar selecci\u00F3n" })] })] })), isExpanded && ((0, jsx_runtime_1.jsx)(framer_motion_1.motion.div, { initial: { opacity: 0, height: 0 }, animate: { opacity: 1, height: 'auto' }, exit: { opacity: 0, height: 0 }, children: viewMode === 'table' ? ((0, jsx_runtime_1.jsx)("div", { className: "p-4 max-h-96 overflow-y-auto", children: (0, jsx_runtime_1.jsx)("div", { className: "overflow-x-auto", children: (0, jsx_runtime_1.jsxs)("table", { className: "w-full text-sm", children: [(0, jsx_runtime_1.jsx)("thead", { className: "bg-gray-50 border-b border-gray-200", children: (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left", children: (0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.size === filteredAndSortedAgents.length && filteredAndSortedAgents.length > 0, onChange: handleSelectAll, className: "rounded border-gray-300" }) }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Nombre" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Estado" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Tipo" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Ejecuciones" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "\u00C9xito" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Cr\u00E9ditos" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Acciones" })] }) }), (0, jsx_runtime_1.jsx)("tbody", { className: "divide-y divide-gray-200", children: filteredAndSortedAgents.map(function (agent) { return ((0, jsx_runtime_1.jsxs)("tr", { className: (0, cn_1.cn)("hover:bg-gray-50 transition-colors", selectedAgents.has(agent.id) && "bg-blue-50"), children: [(0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: (0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.has(agent.id), onChange: function () { return handleSelectAgent(agent.id); }, className: "rounded border-gray-300" }) }), (0, jsx_runtime_1.jsxs)("td", { className: "px-3 py-2", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-medium text-gray-900", children: agent.name }), agent.description && ((0, jsx_runtime_1.jsx)("div", { className: "text-xs text-gray-500 truncate max-w-xs", children: agent.description }))] }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: (0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("px-2 py-0.5 text-xs font-medium rounded-full", agent.isActive
                                                        ? "bg-green-100 text-green-700"
                                                        : "bg-gray-100 text-gray-700"), children: agent.isActive ? 'Activo' : 'Inactivo' }) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2 text-xs text-gray-600", children: agent.config.taskType.replace(new RegExp('_', 'g'), ' ').replace(new RegExp('\\b\\w', 'g'), function (l) { return l.toUpperCase(); }) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2 text-xs text-gray-600", children: agent.stats.totalExecutions }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: agent.stats.totalExecutions > 0 ? (function () {
                                                    var successRate = calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions);
                                                    return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-1", children: [(0, jsx_runtime_1.jsxs)("span", { className: "text-xs font-medium text-gray-900", children: [successRate.toFixed(1), "%"] }), (0, jsx_runtime_1.jsx)("div", { className: "w-12 bg-gray-200 rounded-full h-1", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-1 rounded-full", successRate >= 80 ? "bg-green-500" :
                                                                        successRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: {
                                                                        width: "".concat(successRate, "%")
                                                                    } }) })] }));
                                                })() : ((0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-400", children: "-" })) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2 text-xs text-gray-600", children: agent.stats.creditsUsed.toFixed(2) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: (0, jsx_runtime_1.jsx)(TableActionButton, { agent: agent, onUpdate: fetchAgents, onActivity: function (activity) {
                                                        setRecentActivity(function (prev) { return __spreadArray([activity], prev, true).slice(0, 10); });
                                                    } }) })] }, agent.id)); }) })] }) }) })) : ((0, jsx_runtime_1.jsx)("div", { className: "p-4 space-y-3 max-h-96 overflow-y-auto", children: filteredAndSortedAgents.length === 0 ? ((0, jsx_runtime_1.jsxs)("div", { className: "text-center py-8", children: [(0, jsx_runtime_1.jsx)("p", { className: "text-sm text-gray-500 mb-2", children: searchQuery || filterStatus !== 'all'
                                    ? 'No se encontraron agentes con los filtros aplicados'
                                    : 'No hay agentes configurados' }), searchQuery || filterStatus !== 'all' ? ((0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                    setSearchQuery('');
                                    setFilterStatus('all');
                                }, className: "text-xs text-blue-600 hover:text-blue-700", children: "Limpiar filtros" })) : ((0, jsx_runtime_1.jsxs)(link_1.default, { href: "/continuous-agent", className: "text-xs text-blue-600 hover:text-blue-700 inline-flex items-center gap-1", children: ["Crear primer agente", (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5l7 7-7 7" }) })] }))] })) : ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [activeAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-gray-500 uppercase tracking-wide flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "w-2 h-2 bg-green-500 rounded-full" }), "Activos (", activeAgents.length, ")"] }), activeAgents.map(function (agent) { return ((0, jsx_runtime_1.jsxs)("div", { className: "relative", children: [viewMode !== 'compact' && ((0, jsx_runtime_1.jsxs)("div", { className: "absolute top-3 left-3 z-10 flex flex-col gap-1", children: [(0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.has(agent.id), onChange: function () { return handleSelectAgent(agent.id); }, className: "rounded border-gray-300" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                            handleCompareAgent(agent.id);
                                                            if (comparingAgents.size === 0)
                                                                setShowComparison(true);
                                                        }, className: (0, cn_1.cn)("p-1 rounded transition-colors", comparingAgents.has(agent.id)
                                                            ? "bg-purple-100 text-purple-700"
                                                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"), title: "Comparar agente", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }) })] })), (0, jsx_runtime_1.jsx)(AgentCard, { agent: agent, onUpdate: fetchAgents, viewMode: viewMode, onActivity: function (activity) {
                                                    setRecentActivity(function (prev) { return __spreadArray([activity], prev, true).slice(0, 10); });
                                                } })] }, agent.id)); })] })), inactiveAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [activeAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-gray-500 uppercase tracking-wide mt-4 flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "w-2 h-2 bg-gray-400 rounded-full" }), "Inactivos (", inactiveAgents.length, ")"] })), inactiveAgents.map(function (agent) { return ((0, jsx_runtime_1.jsxs)("div", { className: "relative", children: [viewMode !== 'compact' && ((0, jsx_runtime_1.jsxs)("div", { className: "absolute top-3 left-3 z-10 flex flex-col gap-1", children: [(0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.has(agent.id), onChange: function () { return handleSelectAgent(agent.id); }, className: "rounded border-gray-300" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                            handleCompareAgent(agent.id);
                                                            if (comparingAgents.size === 0)
                                                                setShowComparison(true);
                                                        }, className: (0, cn_1.cn)("p-1 rounded transition-colors", comparingAgents.has(agent.id)
                                                            ? "bg-purple-100 text-purple-700"
                                                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"), title: "Comparar agente", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }) })] })), (0, jsx_runtime_1.jsx)(AgentCard, { agent: agent, onUpdate: fetchAgents, viewMode: viewMode, onActivity: function (activity) {
                                                    setRecentActivity(function (prev) { return __spreadArray([activity], prev, true).slice(0, 10); });
                                                } })] }, agent.id)); })] }))] })) })) }))] }));
}
// Componente para botón de acción en tabla
function TableActionButton(_a) {
    var _this = this;
    var agent = _a.agent, onUpdate = _a.onUpdate, onActivity = _a.onActivity;
    var _b = (0, react_1.useState)(false), isToggling = _b[0], setIsToggling = _b[1];
    var handleToggle = function () { return __awaiter(_this, void 0, void 0, function () {
        var response, action, error_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    setIsToggling(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, fetch("/api/continuous-agent/".concat(agent.id), {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ isActive: !agent.isActive }),
                        })];
                case 2:
                    response = _a.sent();
                    if (response.ok) {
                        action = agent.isActive ? 'deactivated' : 'activated';
                        sonner_1.toast.success(agent.isActive ? 'Agente pausado' : 'Agente activado', {
                            description: "".concat(agent.name, " ha sido ").concat(agent.isActive ? 'pausado' : 'activado', " correctamente"),
                        });
                        if (onActivity) {
                            onActivity({
                                agentId: agent.id,
                                agentName: agent.name,
                                action: action,
                                timestamp: new Date(),
                            });
                        }
                        if (onUpdate) {
                            onUpdate();
                        }
                    }
                    else {
                        sonner_1.toast.error('Error al actualizar agente', {
                            description: 'No se pudo cambiar el estado del agente',
                        });
                    }
                    return [3 /*break*/, 5];
                case 3:
                    error_3 = _a.sent();
                    console.error('Error toggling agent:', error_3);
                    sonner_1.toast.error('Error al actualizar agente', {
                        description: 'Ocurrió un error al cambiar el estado',
                    });
                    return [3 /*break*/, 5];
                case 4:
                    setIsToggling(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    return ((0, jsx_runtime_1.jsx)("button", { onClick: handleToggle, disabled: isToggling, className: (0, cn_1.cn)("px-2 py-1 text-xs font-medium rounded transition-colors", agent.isActive
            ? "bg-red-100 text-red-700 hover:bg-red-200"
            : "bg-green-100 text-green-700 hover:bg-green-200", isToggling && "opacity-50 cursor-not-allowed"), children: isToggling ? '...' : agent.isActive ? 'Pausar' : 'Activar' }));
}
function AgentCard(_a) {
    var _this = this;
    var agent = _a.agent, onUpdate = _a.onUpdate, _b = _a.viewMode, viewMode = _b === void 0 ? 'detailed' : _b, onActivity = _a.onActivity;
    var _c = (0, react_1.useState)(false), isToggling = _c[0], setIsToggling = _c[1];
    var handleToggle = function () { return __awaiter(_this, void 0, void 0, function () {
        var response, action, error_4;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    setIsToggling(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, fetch("/api/continuous-agent/".concat(agent.id), {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ isActive: !agent.isActive }),
                        })];
                case 2:
                    response = _a.sent();
                    if (response.ok) {
                        action = agent.isActive ? 'deactivated' : 'activated';
                        sonner_1.toast.success(agent.isActive ? 'Agente pausado' : 'Agente activado', {
                            description: "".concat(agent.name, " ha sido ").concat(agent.isActive ? 'pausado' : 'activado', " correctamente"),
                        });
                        // Registrar actividad
                        if (onActivity) {
                            onActivity({
                                agentId: agent.id,
                                agentName: agent.name,
                                action: action,
                                timestamp: new Date(),
                            });
                        }
                        // Notificar al componente padre para refrescar
                        if (onUpdate) {
                            onUpdate();
                        }
                    }
                    else {
                        sonner_1.toast.error('Error al actualizar agente', {
                            description: 'No se pudo cambiar el estado del agente',
                        });
                    }
                    return [3 /*break*/, 5];
                case 3:
                    error_4 = _a.sent();
                    console.error('Error toggling agent:', error_4);
                    sonner_1.toast.error('Error al actualizar agente', {
                        description: 'Ocurrió un error al cambiar el estado',
                    });
                    return [3 /*break*/, 5];
                case 4:
                    setIsToggling(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var successRate = calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions);
    if (viewMode === 'compact') {
        return ((0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("p-2 rounded-lg border transition-all hover:shadow-sm", agent.isActive
                ? "bg-green-50 border-green-200"
                : "bg-gray-50 border-gray-200"), children: (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 flex-1 min-w-0", children: [(0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("w-2 h-2 rounded-full flex-shrink-0", agent.isActive ? "bg-green-500 animate-pulse" : "bg-gray-400") }), (0, jsx_runtime_1.jsx)("h5", { className: "font-medium text-gray-900 truncate text-sm", children: agent.name }), (0, jsx_runtime_1.jsxs)("span", { className: "text-xs text-gray-500", children: [agent.stats.totalExecutions, " ejec."] })] }), (0, jsx_runtime_1.jsx)("button", { onClick: handleToggle, disabled: isToggling, className: (0, cn_1.cn)("px-2 py-1 text-xs font-medium rounded transition-colors flex-shrink-0", agent.isActive
                            ? "bg-red-100 text-red-700 hover:bg-red-200"
                            : "bg-green-100 text-green-700 hover:bg-green-200", isToggling && "opacity-50 cursor-not-allowed"), children: isToggling ? '...' : agent.isActive ? 'Pausar' : 'Activar' })] }) }));
    }
    return ((0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("p-3 rounded-lg border transition-all hover:shadow-sm", agent.isActive
            ? "bg-green-50 border-green-200"
            : "bg-gray-50 border-gray-200"), children: [(0, jsx_runtime_1.jsx)("div", { className: "flex items-start justify-between gap-2", children: (0, jsx_runtime_1.jsxs)("div", { className: "flex-1 min-w-0", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 flex-1 min-w-0", children: [(0, jsx_runtime_1.jsx)("h5", { className: "font-medium text-gray-900 truncate", children: agent.name }), (0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("w-2 h-2 rounded-full flex-shrink-0 animate-pulse", agent.isActive ? "bg-green-500" : "bg-gray-400"), title: agent.isActive ? "Activo" : "Inactivo" })] }), (0, jsx_runtime_1.jsx)("button", { onClick: handleToggle, disabled: isToggling, className: (0, cn_1.cn)("ml-2 px-2 py-1 text-xs font-medium rounded transition-colors flex-shrink-0", agent.isActive
                                        ? "bg-red-100 text-red-700 hover:bg-red-200"
                                        : "bg-green-100 text-green-700 hover:bg-green-200", isToggling && "opacity-50 cursor-not-allowed"), title: agent.isActive ? "Desactivar" : "Activar", children: isToggling ? ((0, jsx_runtime_1.jsxs)("svg", { className: "w-3 h-3 animate-spin", fill: "none", viewBox: "0 0 24 24", children: [(0, jsx_runtime_1.jsx)("circle", { className: "opacity-25", cx: "12", cy: "12", r: "10", stroke: "currentColor", strokeWidth: "4" }), (0, jsx_runtime_1.jsx)("path", { className: "opacity-75", fill: "currentColor", d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" })] })) : agent.isActive ? ('Pausar') : ('Activar') })] }), agent.description && ((0, jsx_runtime_1.jsx)("p", { className: "text-sm text-gray-600 mb-2 overflow-hidden", children: agent.description.length > 100
                                ? "".concat(agent.description.substring(0, 100), "...")
                                : agent.description })), (0, jsx_runtime_1.jsxs)("div", { className: "flex flex-wrap items-center gap-3 text-xs text-gray-500", children: [(0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1", title: "Tipo de tarea", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" }) }), agent.config.taskType.replace(new RegExp('_', 'g'), ' ').replace(new RegExp('\\b\\w', 'g'), function (l) { return l.toUpperCase(); })] }), (0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1", title: "Frecuencia: cada ".concat(agent.config.frequency, "s"), children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" }) }), agent.config.frequency, "s"] }), (0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1", title: "Total de ejecuciones", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 10V3L4 14h7v7l9-11h-7z" }) }), agent.stats.totalExecutions] }), agent.stats.successfulExecutions > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-green-600", title: "Ejecuciones exitosas", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M5 13l4 4L19 7" }) }), agent.stats.successfulExecutions] })), agent.stats.failedExecutions > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-red-600", title: "Ejecuciones fallidas", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M6 18L18 6M6 6l12 12" }) }), agent.stats.failedExecutions] })), agent.stats.creditsUsed > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-blue-600", title: "Cr\u00E9ditos usados", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" }) }), agent.stats.creditsUsed.toFixed(2)] })), agent.stripeCreditsRemaining !== null && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-purple-600", title: "Cr\u00E9ditos restantes", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" }) }), agent.stripeCreditsRemaining.toFixed(2)] }))] }), agent.stats.lastExecutionAt && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-2 text-xs text-gray-400", children: ["\u00DAltima ejecuci\u00F3n: ", new Date(agent.stats.lastExecutionAt).toLocaleString('es-ES', {
                                    day: '2-digit',
                                    month: 'short',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })] })), agent.stats.nextExecutionAt && agent.isActive && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-1 text-xs text-gray-400", children: ["Pr\u00F3xima ejecuci\u00F3n: ", new Date(agent.stats.nextExecutionAt).toLocaleString('es-ES', {
                                    day: '2-digit',
                                    month: 'short',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })] })), agent.stats.averageExecutionTime > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-1 text-xs text-gray-400 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 10V3L4 14h7v7l9-11h-7z" }) }), "Tiempo promedio: ", agent.stats.averageExecutionTime.toFixed(2), "s"] })), agent.stats.totalExecutions > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-2 pt-2 border-t border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs mb-1", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-500", children: "Tasa de \u00E9xito" }), (0, jsx_runtime_1.jsxs)("span", { className: "font-medium text-gray-900", children: [successRate.toFixed(1), "%"] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-1", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-1 rounded-full transition-all", successRate >= 80 ? "bg-green-500" : successRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: { width: "".concat(successRate, "%") } }) })] })), agent.config.goal && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-2 pt-2 border-t border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-xs text-gray-500 mb-1", children: "Objetivo:" }), (0, jsx_runtime_1.jsx)("p", { className: "text-xs text-gray-700 italic overflow-hidden", style: {
                                        display: '-webkit-box',
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: 'vertical',
                                        maxHeight: '2.5em'
                                    }, children: agent.config.goal.length > 120
                                        ? "".concat(agent.config.goal.substring(0, 120), "...")
                                        : agent.config.goal })] }))] }) }), (0, jsx_runtime_1.jsxs)("div", { className: "mt-2 flex items-center justify-between gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [agent.stats.totalExecutions > 10 && ((0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("px-2 py-0.5 text-xs font-medium rounded-full", successRate >= 90 ? "bg-green-100 text-green-700" :
                                    successRate >= 70 ? "bg-yellow-100 text-yellow-700" :
                                        "bg-red-100 text-red-700"), children: successRate >= 90 ? "⭐ Excelente" :
                                    successRate >= 70 ? "✓ Bueno" : "⚠ Mejorable" })), agent.isActive && agent.stats.nextExecutionAt && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700", children: ["Pr\u00F3ximo: ", (function () {
                                        var next = new Date(agent.stats.nextExecutionAt);
                                        var now = new Date();
                                        var diffMinutes = Math.floor((next.getTime() - now.getTime()) / 1000 / 60);
                                        if (diffMinutes < 1)
                                            return 'Ahora';
                                        if (diffMinutes < 60)
                                            return "".concat(diffMinutes, "m");
                                        var hours = Math.floor(diffMinutes / 60);
                                        return "".concat(hours, "h");
                                    })()] }))] }), (0, jsx_runtime_1.jsxs)(link_1.default, { href: "/continuous-agent", className: "text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1", children: ["Ver detalles", (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5l7 7-7 7" }) })] })] })] }));
}
function AgentsSection(_a) {
    var _this = this;
    var className = _a.className;
    var _b = (0, react_1.useState)([]), agents = _b[0], setAgents = _b[1];
    var _c = (0, react_1.useState)(true), isLoading = _c[0], setIsLoading = _c[1];
    var _d = (0, react_1.useState)(false), isExpanded = _d[0], setIsExpanded = _d[1];
    var _e = (0, react_1.useState)(''), searchQuery = _e[0], setSearchQuery = _e[1];
    var _f = (0, react_1.useState)('all'), filterStatus = _f[0], setFilterStatus = _f[1];
    var _g = (0, react_1.useState)('name'), sortBy = _g[0], setSortBy = _g[1];
    var _h = (0, react_1.useState)('asc'), sortOrder = _h[0], setSortOrder = _h[1];
    var _j = (0, react_1.useState)('detailed'), viewMode = _j[0], setViewMode = _j[1];
    var _k = (0, react_1.useState)(false), showCharts = _k[0], setShowCharts = _k[1];
    var _l = (0, react_1.useState)(null), selectedAgent = _l[0], setSelectedAgent = _l[1];
    var _m = (0, react_1.useState)(new Set()), selectedAgents = _m[0], setSelectedAgents = _m[1];
    var _o = (0, react_1.useState)([]), recentActivity = _o[0], setRecentActivity = _o[1];
    var _p = (0, react_1.useState)(false), showBulkActions = _p[0], setShowBulkActions = _p[1];
    var _q = (0, react_1.useState)('all'), filterTaskType = _q[0], setFilterTaskType = _q[1];
    var _r = (0, react_1.useState)('all'), filterCredits = _r[0], setFilterCredits = _r[1];
    var _s = (0, react_1.useState)(new Set()), comparingAgents = _s[0], setComparingAgents = _s[1];
    var _t = (0, react_1.useState)(false), showComparison = _t[0], setShowComparison = _t[1];
    var _u = (0, react_1.useState)(false), showAlerts = _u[0], setShowAlerts = _u[1];
    var _v = (0, react_1.useState)([]), savedFilters = _v[0], setSavedFilters = _v[1];
    var _w = (0, react_1.useState)(false), showExecutionCalendar = _w[0], setShowExecutionCalendar = _w[1];
    var _x = (0, react_1.useState)(false), showTrends = _x[0], setShowTrends = _x[1];
    var _y = (0, react_1.useState)('7d'), selectedTimeRange = _y[0], setSelectedTimeRange = _y[1];
    var fetchAgents = function () { return __awaiter(_this, void 0, void 0, function () {
        var response, data, error_5;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 4, 5, 6]);
                    return [4 /*yield*/, fetch('/api/continuous-agent')];
                case 1:
                    response = _a.sent();
                    if (!response.ok) return [3 /*break*/, 3];
                    return [4 /*yield*/, response.json()];
                case 2:
                    data = _a.sent();
                    setAgents(Array.isArray(data) ? data : []);
                    _a.label = 3;
                case 3: return [3 /*break*/, 6];
                case 4:
                    error_5 = _a.sent();
                    console.error('Error fetching agents:', error_5);
                    return [3 /*break*/, 6];
                case 5:
                    setIsLoading(false);
                    return [7 /*endfinally*/];
                case 6: return [2 /*return*/];
            }
        });
    }); };
    (0, react_1.useEffect)(function () {
        fetchAgents();
        // Refresh every 10 seconds
        var interval = setInterval(fetchAgents, 10000);
        return function () { return clearInterval(interval); };
    }, []);
    // Detectar cambios en agentes para actividad reciente
    (0, react_1.useEffect)(function () {
        agents.forEach(function (agent) {
            // Esto se puede mejorar con un sistema de tracking más sofisticado
            // Por ahora, solo trackeamos cuando se detectan ejecuciones recientes
            if (agent.stats.lastExecutionAt) {
                var lastExec_2 = new Date(agent.stats.lastExecutionAt);
                var now = new Date();
                var diffMinutes = (now.getTime() - lastExec_2.getTime()) / 1000 / 60;
                if (diffMinutes < 5) {
                    setRecentActivity(function (prev) {
                        var exists = prev.some(function (a) {
                            return a.agentId === agent.id &&
                                a.action === 'executed' &&
                                Math.abs(a.timestamp.getTime() - lastExec_2.getTime()) < 60000;
                        });
                        if (!exists) {
                            return __spreadArray([{
                                    agentId: agent.id,
                                    agentName: agent.name,
                                    action: 'executed',
                                    timestamp: lastExec_2,
                                }], prev, true).slice(0, 10);
                        }
                        return prev;
                    });
                }
            }
        });
    }, [agents]);
    // Estadísticas agregadas
    var totalStats = (0, react_1.useMemo)(function () {
        return agents.reduce(function (acc, agent) { return ({
            totalExecutions: acc.totalExecutions + agent.stats.totalExecutions,
            successfulExecutions: acc.successfulExecutions + agent.stats.successfulExecutions,
            failedExecutions: acc.failedExecutions + agent.stats.failedExecutions,
            creditsUsed: acc.creditsUsed + agent.stats.creditsUsed,
            totalAgents: acc.totalAgents + 1,
        }); }, {
            totalExecutions: 0,
            successfulExecutions: 0,
            failedExecutions: 0,
            creditsUsed: 0,
            totalAgents: 0,
        });
    }, [agents]);
    // Filtrar y ordenar agentes
    var filteredAndSortedAgents = (0, react_1.useMemo)(function () {
        var filtered = agents;
        // Filtro por estado
        if (filterStatus === 'active') {
            filtered = filtered.filter(function (a) { return a.isActive; });
        }
        else if (filterStatus === 'inactive') {
            filtered = filtered.filter(function (a) { return !a.isActive; });
        }
        // Filtro por búsqueda
        if (searchQuery.trim()) {
            var query_2 = searchQuery.toLowerCase();
            filtered = filtered.filter(function (agent) {
                return agent.name.toLowerCase().includes(query_2) ||
                    agent.description.toLowerCase().includes(query_2) ||
                    agent.config.taskType.toLowerCase().includes(query_2);
            });
        }
        // Filtro por tipo de tarea
        if (filterTaskType !== 'all') {
            filtered = filtered.filter(function (agent) { return agent.config.taskType === filterTaskType; });
        }
        // Filtro por créditos
        if (filterCredits === 'with') {
            filtered = filtered.filter(function (agent) { return agent.stats.creditsUsed > 0; });
        }
        else if (filterCredits === 'without') {
            filtered = filtered.filter(function (agent) { return agent.stats.creditsUsed === 0; });
        }
        // Ordenamiento
        var sorted = __spreadArray([], filtered, true).sort(function (a, b) {
            var comparison = 0;
            switch (sortBy) {
                case 'name':
                    comparison = a.name.localeCompare(b.name);
                    break;
                case 'status':
                    comparison = Number(b.isActive) - Number(a.isActive);
                    break;
                case 'executions':
                    comparison = b.stats.totalExecutions - a.stats.totalExecutions;
                    break;
                case 'success':
                    var aRate = a.stats.totalExecutions > 0
                        ? a.stats.successfulExecutions / a.stats.totalExecutions
                        : 0;
                    var bRate = b.stats.totalExecutions > 0
                        ? b.stats.successfulExecutions / b.stats.totalExecutions
                        : 0;
                    comparison = bRate - aRate;
                    break;
                case 'created':
                    comparison = new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
                    break;
            }
            return sortOrder === 'asc' ? comparison : -comparison;
        });
        return sorted;
    }, [agents, filterStatus, searchQuery, sortBy, sortOrder, filterTaskType, filterCredits]);
    var activeAgents = filteredAndSortedAgents.filter(function (a) { return a.isActive; });
    var inactiveAgents = filteredAndSortedAgents.filter(function (a) { return !a.isActive; });
    // Tipos de tarea únicos para filtro
    var taskTypes = (0, react_1.useMemo)(function () {
        var types = new Set(agents.map(function (a) { return a.config.taskType; }));
        return Array.from(types).sort();
    }, [agents]);
    // Acciones en lote
    var handleBulkToggle = function (activate) { return __awaiter(_this, void 0, void 0, function () {
        var promises, results, successful, failed, error_6;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (selectedAgents.size === 0) {
                        sonner_1.toast.error('No hay agentes seleccionados', {
                            description: 'Selecciona al menos un agente para realizar la acción',
                        });
                        return [2 /*return*/];
                    }
                    promises = Array.from(selectedAgents).map(function (agentId) {
                        return fetch("/api/continuous-agent/".concat(agentId), {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ isActive: activate }),
                        });
                    });
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, Promise.allSettled(promises)];
                case 2:
                    results = _a.sent();
                    successful = results.filter(function (r) { return r.status === 'fulfilled'; }).length;
                    failed = results.length - successful;
                    if (successful > 0) {
                        sonner_1.toast.success("".concat(successful, " agente(s) ").concat(activate ? 'activado(s)' : 'pausado(s)'), {
                            description: failed > 0 ? "".concat(failed, " fallaron") : undefined,
                        });
                        setSelectedAgents(new Set());
                        fetchAgents();
                    }
                    else {
                        sonner_1.toast.error('Error al actualizar agentes', {
                            description: 'No se pudo cambiar el estado de los agentes',
                        });
                    }
                    return [3 /*break*/, 4];
                case 3:
                    error_6 = _a.sent();
                    sonner_1.toast.error('Error al actualizar agentes', {
                        description: 'Ocurrió un error al procesar la acción',
                    });
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    }); };
    var handleSelectAll = function () {
        if (selectedAgents.size === filteredAndSortedAgents.length) {
            setSelectedAgents(new Set());
        }
        else {
            setSelectedAgents(new Set(filteredAndSortedAgents.map(function (a) { return a.id; })));
        }
    };
    var handleSelectAgent = function (agentId) {
        setSelectedAgents(function (prev) {
            var newSet = new Set(prev);
            if (newSet.has(agentId)) {
                newSet.delete(agentId);
            }
            else {
                newSet.add(agentId);
            }
            return newSet;
        });
    };
    var handleCompareAgent = function (agentId) {
        setComparingAgents(function (prev) {
            var newSet = new Set(prev);
            if (newSet.has(agentId)) {
                newSet.delete(agentId);
            }
            else {
                if (newSet.size < 3) {
                    newSet.add(agentId);
                }
                else {
                    sonner_1.toast.error('Máximo 3 agentes para comparar', {
                        description: 'Deselecciona un agente antes de agregar otro',
                    });
                }
            }
            return newSet;
        });
    };
    var handleSaveFilter = function () {
        var name = prompt('Nombre para este filtro:');
        if (name) {
            setSavedFilters(function (prev) { return __spreadArray(__spreadArray([], prev, true), [{
                    name: name,
                    filters: {
                        searchQuery: searchQuery,
                        filterStatus: filterStatus,
                        filterTaskType: filterTaskType,
                        filterCredits: filterCredits,
                    },
                }], false); });
            sonner_1.toast.success('Filtro guardado', {
                description: "Filtro \"".concat(name, "\" guardado correctamente"),
            });
        }
    };
    var handleLoadFilter = function (filter) {
        setSearchQuery(filter.filters.searchQuery);
        setFilterStatus(filter.filters.filterStatus);
        setFilterTaskType(filter.filters.filterTaskType);
        setFilterCredits(filter.filters.filterCredits);
        sonner_1.toast.success('Filtro aplicado', {
            description: "Filtro \"".concat(filter.name, "\" aplicado"),
        });
    };
    // Distribución por tipo de tarea
    var taskTypeDistribution = (0, react_1.useMemo)(function () {
        var distribution = {};
        agents.forEach(function (agent) {
            var type = agent.config.taskType;
            distribution[type] = (distribution[type] || 0) + 1;
        });
        return distribution;
    }, [agents]);
    // Agentes con mejor rendimiento
    var topPerformers = (0, react_1.useMemo)(function () {
        return __spreadArray([], agents, true).filter(function (a) { return a.stats.totalExecutions > 0; })
            .map(function (a) { return ({
            agent: a,
            successRate: (a.stats.successfulExecutions / a.stats.totalExecutions) * 100,
        }); })
            .sort(function (a, b) { return b.successRate - a.successRate; })
            .slice(0, 3);
    }, [agents]);
    // Agentes más activos
    var mostActive = (0, react_1.useMemo)(function () {
        return __spreadArray([], agents, true).filter(function (a) { return a.isActive; })
            .sort(function (a, b) { return b.stats.totalExecutions - a.stats.totalExecutions; })
            .slice(0, 3);
    }, [agents]);
    // Alertas y notificaciones
    var alerts = (0, react_1.useMemo)(function () {
        var alertsList = [];
        agents.forEach(function (agent) {
            // Agentes inactivos por mucho tiempo
            if (!agent.isActive && agent.stats.lastExecutionAt) {
                var lastExec = new Date(agent.stats.lastExecutionAt);
                var daysSince = (Date.now() - lastExec.getTime()) / (1000 * 60 * 60 * 24);
                if (daysSince > 7) {
                    alertsList.push({
                        type: 'info',
                        message: "Inactivo por ".concat(Math.floor(daysSince), " d\u00EDas"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
            }
            // Alta tasa de fallos
            if (agent.stats.totalExecutions > 10) {
                var failureRate = (agent.stats.failedExecutions / agent.stats.totalExecutions) * 100;
                if (failureRate > 50) {
                    alertsList.push({
                        type: 'error',
                        message: "Tasa de fallos alta: ".concat(failureRate.toFixed(1), "%"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
                else if (failureRate > 30) {
                    alertsList.push({
                        type: 'warning',
                        message: "Tasa de fallos moderada: ".concat(failureRate.toFixed(1), "%"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
            }
            // Sin créditos
            if (agent.stripeCreditsRemaining !== null && agent.stripeCreditsRemaining < 1) {
                alertsList.push({
                    type: 'warning',
                    message: 'Sin créditos disponibles',
                    agentId: agent.id,
                    agentName: agent.name,
                });
            }
            // Sin ejecuciones recientes en agentes activos
            if (agent.isActive && agent.stats.lastExecutionAt) {
                var lastExec = new Date(agent.stats.lastExecutionAt);
                var hoursSince = (Date.now() - lastExec.getTime()) / (1000 * 60 * 60);
                var expectedExecutions = Math.floor(hoursSince / (agent.config.frequency / 3600));
                if (expectedExecutions > 3 && agent.stats.totalExecutions > 0) {
                    alertsList.push({
                        type: 'warning',
                        message: "Sin ejecuciones en ".concat(Math.floor(hoursSince), " horas"),
                        agentId: agent.id,
                        agentName: agent.name,
                    });
                }
            }
        });
        return alertsList;
    }, [agents]);
    // Agentes para comparación
    var agentsToCompare = (0, react_1.useMemo)(function () {
        return Array.from(comparingAgents)
            .map(function (id) { return agents.find(function (a) { return a.id === id; }); })
            .filter(function (a) { return a !== undefined; });
    }, [comparingAgents, agents]);
    // Análisis de prompts
    var promptAnalytics = (0, react_1.useMemo)(function () {
        var agentsWithGoals = agents.filter(function (a) { return a.config.goal && a.config.goal.trim().length > 0; });
        var totalGoalsLength = agentsWithGoals.reduce(function (sum, a) { var _a; return sum + (((_a = a.config.goal) === null || _a === void 0 ? void 0 : _a.length) || 0); }, 0);
        var averageGoalLength = agentsWithGoals.length > 0 ? totalGoalsLength / agentsWithGoals.length : 0;
        // Análisis de salud de prompts
        var healthyGoals = agentsWithGoals.filter(function (a) {
            var goal = a.config.goal || '';
            return goal.length >= 50 && goal.length <= 1000;
        });
        var healthScore = agentsWithGoals.length > 0
            ? Math.round((healthyGoals.length / agentsWithGoals.length) * 100)
            : 0;
        return {
            agentsWithGoals: agentsWithGoals.length,
            totalAgents: agents.length,
            averageGoalLength: Math.round(averageGoalLength),
            healthyGoals: healthyGoals.length,
            healthScore: healthScore,
        };
    }, [agents]);
    // Análisis de rendimiento por tipo de tarea
    var taskTypePerformance = (0, react_1.useMemo)(function () {
        var performance = {};
        agents.forEach(function (agent) {
            var type = agent.config.taskType;
            if (!performance[type]) {
                performance[type] = {
                    count: 0,
                    totalExecutions: 0,
                    successfulExecutions: 0,
                    averageSuccessRate: 0,
                };
            }
            performance[type].count++;
            performance[type].totalExecutions += agent.stats.totalExecutions;
            performance[type].successfulExecutions += agent.stats.successfulExecutions;
        });
        // Calcular tasa promedio de éxito por tipo
        Object.keys(performance).forEach(function (type) {
            var perf = performance[type];
            perf.averageSuccessRate = perf.totalExecutions > 0
                ? (perf.successfulExecutions / perf.totalExecutions) * 100
                : 0;
        });
        return performance;
    }, [agents]);
    // Función para exportar agentes
    var handleExport = function () {
        var dataStr = JSON.stringify(filteredAndSortedAgents, null, 2);
        var dataBlob = new Blob([dataStr], { type: 'application/json' });
        var url = URL.createObjectURL(dataBlob);
        var link = document.createElement('a');
        link.href = url;
        link.download = "agentes-".concat(new Date().toISOString().split('T')[0], ".json");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        sonner_1.toast.success('Agentes exportados', {
            description: "".concat(filteredAndSortedAgents.length, " agente(s) exportado(s) correctamente"),
        });
    };
    // Función para exportar CSV
    var handleExportCSV = function () {
        var headers = ['Nombre', 'Estado', 'Tipo', 'Ejecuciones', 'Exitosas', 'Fallidas', 'Tasa Éxito', 'Créditos Usados', 'Creado'];
        var rows = filteredAndSortedAgents.map(function (agent) {
            var successRate = agent.stats.totalExecutions > 0
                ? ((agent.stats.successfulExecutions / agent.stats.totalExecutions) * 100).toFixed(2)
                : '0.00';
            return [
                agent.name,
                agent.isActive ? 'Activo' : 'Inactivo',
                agent.config.taskType,
                agent.stats.totalExecutions.toString(),
                agent.stats.successfulExecutions.toString(),
                agent.stats.failedExecutions.toString(),
                "".concat(successRate, "%"),
                agent.stats.creditsUsed.toFixed(2),
                new Date(agent.createdAt).toLocaleDateString('es-ES'),
            ];
        });
        var csvContent = __spreadArray([
            headers.join(',')
        ], rows.map(function (row) { return row.map(function (cell) { return "\"".concat(cell, "\""); }).join(','); }), true).join('\n');
        var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        var url = URL.createObjectURL(blob);
        var link = document.createElement('a');
        link.href = url;
        link.download = "agentes-".concat(new Date().toISOString().split('T')[0], ".csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        sonner_1.toast.success('Agentes exportados a CSV', {
            description: "".concat(filteredAndSortedAgents.length, " agente(s) exportado(s) correctamente"),
        });
    };
    if (isLoading) {
        return ((0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("bg-white border border-gray-200 rounded-lg p-4", className), children: (0, jsx_runtime_1.jsxs)("div", { className: "animate-pulse", children: [(0, jsx_runtime_1.jsx)("div", { className: "h-4 bg-gray-200 rounded w-1/4 mb-3" }), (0, jsx_runtime_1.jsx)("div", { className: "h-3 bg-gray-200 rounded w-1/2" })] }) }));
    }
    if (agents.length === 0) {
        return null;
    }
    return ((0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("bg-white border border-gray-200 rounded-lg shadow-sm", className), children: [(0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 flex items-center justify-between border-b border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-3 flex-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-5 h-5 text-gray-600", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" }) }), (0, jsx_runtime_1.jsx)("h3", { className: "font-semibold text-gray-900", children: "Agentes Continuos" })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [activeAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full", children: [activeAgents.length, " activo", activeAgents.length !== 1 ? 's' : ''] })), (0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-gray-100 text-gray-700 text-xs font-medium rounded-full", children: [filteredAndSortedAgents.length, " / ", agents.length] }), selectedAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full", children: [selectedAgents.size, " seleccionado", selectedAgents.size !== 1 ? 's' : ''] })), comparingAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full", children: [comparingAgents.size, " comparando"] })), alerts.length > 0 && ((0, jsx_runtime_1.jsxs)("button", { onClick: function () { return setShowAlerts(!showAlerts); }, className: (0, cn_1.cn)("px-2 py-0.5 text-xs font-medium rounded-full transition-colors relative", showAlerts
                                            ? "bg-yellow-100 text-yellow-700"
                                            : "bg-red-100 text-red-700 hover:bg-red-200"), children: [alerts.length, " alerta", alerts.length !== 1 ? 's' : '', (0, jsx_runtime_1.jsx)("span", { className: "absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" })] }))] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [isExpanded && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return setShowCharts(!showCharts); }, className: (0, cn_1.cn)("px-2 py-1.5 text-xs rounded-lg transition-colors flex items-center gap-1", showCharts
                                            ? "bg-blue-100 text-blue-700 hover:bg-blue-200"
                                            : "text-gray-600 hover:text-gray-700 hover:bg-gray-100"), title: "Mostrar gr\u00E1ficos", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }) }), (0, jsx_runtime_1.jsxs)("div", { className: "relative group", children: [(0, jsx_runtime_1.jsx)("button", { onClick: handleExport, className: "px-2 py-1.5 text-xs text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors flex items-center gap-1", title: "Exportar agentes", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" }) }) }), (0, jsx_runtime_1.jsxs)("div", { className: "absolute right-0 top-full mt-1 w-40 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50", children: [(0, jsx_runtime_1.jsx)("button", { onClick: handleExport, className: "w-full text-left px-3 py-2 text-xs hover:bg-gray-50 rounded-t-lg", children: "Exportar JSON" }), (0, jsx_runtime_1.jsx)("button", { onClick: handleExportCSV, className: "w-full text-left px-3 py-2 text-xs hover:bg-gray-50 rounded-b-lg", children: "Exportar CSV" })] })] }), (0, jsx_runtime_1.jsx)("div", { className: "relative group", children: (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                var modes = ['detailed', 'compact', 'table'];
                                                var currentIndex = modes.indexOf(viewMode);
                                                setViewMode(modes[(currentIndex + 1) % modes.length]);
                                            }, className: "px-2 py-1.5 text-xs text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors", title: "Vista: ".concat(viewMode === 'compact' ? 'Compacta' : viewMode === 'table' ? 'Tabla' : 'Detallada'), children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: viewMode === 'compact' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" })) : viewMode === 'table' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" })) : ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M4 6h16M4 12h16M4 18h16" })) }) }) }), selectedAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return setShowBulkActions(!showBulkActions); }, className: "px-2 py-1.5 text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 rounded-lg transition-colors", title: "Acciones en lote", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" }) }) }), showBulkActions && ((0, jsx_runtime_1.jsxs)("div", { className: "absolute right-0 top-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 flex gap-1 p-1", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(true); }, className: "px-2 py-1 text-xs bg-green-100 text-green-700 hover:bg-green-200 rounded transition-colors", children: "Activar todos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(false); }, className: "px-2 py-1 text-xs bg-red-100 text-red-700 hover:bg-red-200 rounded transition-colors", children: "Pausar todos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setSelectedAgents(new Set()); }, className: "px-2 py-1 text-xs bg-gray-100 text-gray-700 hover:bg-gray-200 rounded transition-colors", children: "Limpiar" })] }))] }))] })), (0, jsx_runtime_1.jsxs)(link_1.default, { href: "/continuous-agent", className: "px-3 py-1.5 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors flex items-center gap-1", title: "Ver todos los agentes", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" }) }), "Ver todos"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setIsExpanded(!isExpanded); }, className: "p-1.5 hover:bg-gray-100 rounded-lg transition-colors", title: isExpanded ? "Contraer" : "Expandir", children: (0, jsx_runtime_1.jsx)("svg", { className: (0, cn_1.cn)("w-5 h-5 text-gray-500 transition-transform", isExpanded && "transform rotate-180"), fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M19 9l-7 7-7-7" }) }) })] })] }), isExpanded && totalStats.totalAgents > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 bg-gray-50 border-b border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-3 text-xs", children: [(0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Total Ejecuciones" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900 text-lg", children: totalStats.totalExecutions.toLocaleString() })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Exitosas" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-green-600 text-lg", children: totalStats.successfulExecutions.toLocaleString() })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Fallidas" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-red-600 text-lg", children: totalStats.failedExecutions.toLocaleString() })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500 mb-1", children: "Cr\u00E9ditos Usados" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-blue-600 text-lg", children: totalStats.creditsUsed.toFixed(2) })] })] }), totalStats.totalExecutions > 0 && (function () {
                        var globalSuccessRate = calculateSuccessRate(totalStats.successfulExecutions, totalStats.totalExecutions);
                        return ((0, jsx_runtime_1.jsxs)("div", { className: "mt-3 p-2 bg-white rounded-lg border border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs mb-1", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-500 font-medium", children: "Tasa de \u00E9xito global" }), (0, jsx_runtime_1.jsxs)("span", { className: "font-bold text-gray-900 text-sm", children: [globalSuccessRate.toFixed(1), "%"] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-2", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-2 rounded-full transition-all", globalSuccessRate >= 80 ? "bg-green-500" :
                                            globalSuccessRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: {
                                            width: "".concat(globalSuccessRate, "%")
                                        } }) })] }));
                    })(), (0, jsx_runtime_1.jsxs)("div", { className: "mt-2 grid grid-cols-2 md:grid-cols-3 gap-2 text-xs", children: [(0, jsx_runtime_1.jsxs)("div", { className: "p-1.5 bg-white rounded border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Agentes Totales" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900", children: totalStats.totalAgents })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-1.5 bg-white rounded border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Activos" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-green-600", children: activeAgents.length })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-1.5 bg-white rounded border border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Inactivos" }), (0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-600", children: inactiveAgents.length })] })] })] })), isExpanded && showCharts && totalStats.totalAgents > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 bg-white border-b border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4", children: [(0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700", children: "Distribuci\u00F3n por Tipo" }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-1.5", children: Object.entries(taskTypeDistribution).map(function (_a) {
                                            var type = _a[0], count = _a[1];
                                            var percentage = (count / agents.length) * 100;
                                            return ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-600 capitalize", children: type.replace(new RegExp('_', 'g'), ' ') }), (0, jsx_runtime_1.jsxs)("span", { className: "font-medium text-gray-900", children: [count, " (", percentage.toFixed(0), "%)"] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-2", children: (0, jsx_runtime_1.jsx)("div", { className: "bg-blue-500 h-2 rounded-full transition-all", style: { width: "".concat(percentage, "%") } }) })] }, type));
                                        }) })] }), (0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700", children: "Mejor Rendimiento" }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-2", children: topPerformers.length > 0 ? (topPerformers.map(function (_a, index) {
                                            var agent = _a.agent, successRate = _a.successRate;
                                            return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 p-2 bg-gray-50 rounded-lg", children: [(0, jsx_runtime_1.jsx)("div", { className: "flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center text-xs font-bold text-white", children: index + 1 }), (0, jsx_runtime_1.jsxs)("div", { className: "flex-1 min-w-0", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-xs font-medium text-gray-900 truncate", children: agent.name }), (0, jsx_runtime_1.jsxs)("div", { className: "text-xs text-gray-500", children: [successRate.toFixed(1), "% \u00E9xito \u2022 ", agent.stats.totalExecutions, " ejec."] })] }), (0, jsx_runtime_1.jsx)("div", { className: "flex-shrink-0", children: (0, jsx_runtime_1.jsxs)("div", { className: "w-12 h-12 relative", children: [(0, jsx_runtime_1.jsxs)("svg", { className: "transform -rotate-90", viewBox: "0 0 36 36", children: [(0, jsx_runtime_1.jsx)("circle", { cx: "18", cy: "18", r: "16", fill: "none", stroke: "#e5e7eb", strokeWidth: "3" }), (0, jsx_runtime_1.jsx)("circle", { cx: "18", cy: "18", r: "16", fill: "none", stroke: successRate >= 80 ? "#10b981" : successRate >= 50 ? "#f59e0b" : "#ef4444", strokeWidth: "3", strokeDasharray: "".concat(successRate, ", 100"), strokeLinecap: "round" })] }), (0, jsx_runtime_1.jsxs)("div", { className: "absolute inset-0 flex items-center justify-center text-xs font-semibold text-gray-900", children: [successRate.toFixed(0), "%"] })] }) })] }, agent.id));
                                        })) : ((0, jsx_runtime_1.jsx)("p", { className: "text-xs text-gray-500 text-center py-2", children: "No hay datos suficientes" })) })] })] }), mostActive.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-4 pt-4 border-t border-gray-200", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700 mb-2", children: "M\u00E1s Activos" }), (0, jsx_runtime_1.jsx)("div", { className: "flex gap-2", children: mostActive.map(function (agent, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: "flex-1 p-2 bg-gradient-to-br from-green-50 to-blue-50 rounded-lg border border-green-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-xs font-medium text-gray-900 truncate", children: agent.name }), (0, jsx_runtime_1.jsxs)("div", { className: "text-xs text-gray-600 mt-1", children: [agent.stats.totalExecutions, " ejecuciones"] }), (0, jsx_runtime_1.jsxs)("div", { className: "mt-1 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("div", { className: "flex-1 bg-gray-200 rounded-full h-1", children: (function () {
                                                        var _a;
                                                        var maxExecutions = ((_a = mostActive[0]) === null || _a === void 0 ? void 0 : _a.stats.totalExecutions) || 1;
                                                        var executionPercentage = Math.min((agent.stats.totalExecutions / maxExecutions) * 100, 100);
                                                        return ((0, jsx_runtime_1.jsx)("div", { className: "bg-green-500 h-1 rounded-full", style: {
                                                                width: "".concat(executionPercentage, "%")
                                                            } }));
                                                    })() }), (0, jsx_runtime_1.jsxs)("span", { className: "text-xs text-gray-500", children: ["#", index + 1] })] })] }, agent.id)); }) })] })), promptAnalytics.agentsWithGoals > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-4 pt-4 border-t border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-2", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700", children: "Estad\u00EDsticas de Prompts" }), (0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("text-xs font-bold", promptAnalytics.healthScore >= 80 ? "text-green-600" :
                                            promptAnalytics.healthScore >= 60 ? "text-yellow-600" : "text-red-600"), children: ["Salud: ", promptAnalytics.healthScore, "%"] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "grid grid-cols-2 gap-2 text-xs", children: [(0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-gray-50 rounded", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900", children: promptAnalytics.agentsWithGoals }), (0, jsx_runtime_1.jsxs)("div", { className: "text-gray-500", children: ["Con objetivos (", Math.round((promptAnalytics.agentsWithGoals / promptAnalytics.totalAgents) * 100), "%)"] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-gray-50 rounded", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-blue-600", children: promptAnalytics.healthyGoals }), (0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Objetivos saludables" })] }), (0, jsx_runtime_1.jsxs)("div", { className: "p-2 bg-gray-50 rounded col-span-2", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-semibold text-gray-900", children: promptAnalytics.averageGoalLength.toLocaleString() }), (0, jsx_runtime_1.jsx)("div", { className: "text-gray-500", children: "Promedio de caracteres" })] })] })] })), Object.keys(taskTypePerformance).length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-4 pt-4 border-t border-gray-200", children: [(0, jsx_runtime_1.jsx)("h4", { className: "text-xs font-semibold text-gray-700 mb-2", children: "Rendimiento por Tipo" }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-2", children: Object.entries(taskTypePerformance)
                                    .sort(function (a, b) { return b[1].averageSuccessRate - a[1].averageSuccessRate; })
                                    .map(function (_a) {
                                    var type = _a[0], perf = _a[1];
                                    return ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-600 capitalize", children: type.replace(new RegExp('_', 'g'), ' ') }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsxs)("span", { className: "text-gray-500", children: [perf.count, " agente", perf.count !== 1 ? 's' : ''] }), (0, jsx_runtime_1.jsxs)("span", { className: (0, cn_1.cn)("font-medium", perf.averageSuccessRate >= 80 ? "text-green-600" :
                                                                    perf.averageSuccessRate >= 50 ? "text-yellow-600" : "text-red-600"), children: [perf.averageSuccessRate.toFixed(1), "% \u00E9xito"] })] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-1.5", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-1.5 rounded-full transition-all", perf.averageSuccessRate >= 80 ? "bg-green-500" :
                                                        perf.averageSuccessRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: { width: "".concat(Math.min(perf.averageSuccessRate, 100), "%") } }) }), (0, jsx_runtime_1.jsxs)("div", { className: "text-xs text-gray-400", children: [perf.totalExecutions, " ejecuciones totales"] })] }, type));
                                }) })] }))] })), isExpanded && showAlerts && alerts.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 bg-red-50 border-b border-red-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-2", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-red-700 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" }) }), "Alertas (", alerts.length, ")"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setShowAlerts(false); }, className: "text-xs text-red-600 hover:text-red-700", children: "Cerrar" })] }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-1.5 max-h-32 overflow-y-auto", children: alerts.map(function (alert, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("p-2 rounded-lg text-xs flex items-start gap-2", alert.type === 'error' ? "bg-red-100 text-red-800" :
                                alert.type === 'warning' ? "bg-yellow-100 text-yellow-800" :
                                    "bg-blue-100 text-blue-800"), children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4 flex-shrink-0 mt-0.5", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: alert.type === 'error' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" })) : alert.type === 'warning' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" })) : ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" })) }), (0, jsx_runtime_1.jsxs)("div", { className: "flex-1", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-medium", children: alert.agentName || 'Sistema' }), (0, jsx_runtime_1.jsx)("div", { children: alert.message })] })] }, index)); }) })] })), isExpanded && showComparison && agentsToCompare.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-3 bg-purple-50 border-b border-purple-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-3", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-purple-700 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }), "Comparaci\u00F3n (", agentsToCompare.length, ")"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                    setComparingAgents(new Set());
                                    setShowComparison(false);
                                }, className: "text-xs text-purple-600 hover:text-purple-700", children: "Cerrar" })] }), (0, jsx_runtime_1.jsx)("div", { className: "overflow-x-auto", children: (0, jsx_runtime_1.jsxs)("table", { className: "w-full text-xs", children: [(0, jsx_runtime_1.jsx)("thead", { children: (0, jsx_runtime_1.jsxs)("tr", { className: "border-b border-purple-200", children: [(0, jsx_runtime_1.jsx)("th", { className: "px-2 py-1 text-left text-purple-700", children: "M\u00E9trica" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("th", { className: "px-2 py-1 text-left text-purple-700", children: agent.name }, agent.id)); })] }) }), (0, jsx_runtime_1.jsxs)("tbody", { className: "divide-y divide-purple-100", children: [(0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Estado" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1", children: (0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("px-1.5 py-0.5 text-xs rounded-full", agent.isActive ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-700"), children: agent.isActive ? 'Activo' : 'Inactivo' }) }, agent.id)); })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Ejecuciones" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 text-gray-600", children: agent.stats.totalExecutions }, agent.id)); })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Tasa \u00C9xito" }), agentsToCompare.map(function (agent) {
                                                    var rate = calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions);
                                                    return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1", children: (0, jsx_runtime_1.jsx)("div", { className: "flex items-center gap-1", children: (0, jsx_runtime_1.jsxs)("span", { className: (0, cn_1.cn)("text-xs font-medium", rate >= 80 ? "text-green-600" : rate >= 50 ? "text-yellow-600" : "text-red-600"), children: [rate.toFixed(1), "%"] }) }) }, agent.id));
                                                })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Cr\u00E9ditos Usados" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 text-gray-600", children: agent.stats.creditsUsed.toFixed(2) }, agent.id)); })] }), (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 font-medium text-gray-700", children: "Tiempo Promedio" }), agentsToCompare.map(function (agent) { return ((0, jsx_runtime_1.jsx)("td", { className: "px-2 py-1 text-gray-600", children: agent.stats.averageExecutionTime > 0
                                                        ? "".concat(agent.stats.averageExecutionTime.toFixed(2), "s")
                                                        : '-' }, agent.id)); })] })] })] }) })] })), isExpanded && recentActivity.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 bg-blue-50 border-b border-blue-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-2", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-gray-700 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 10V3L4 14h7v7l9-11h-7z" }) }), "Actividad Reciente"] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setRecentActivity([]); }, className: "text-xs text-gray-500 hover:text-gray-700", children: "Limpiar" })] }), (0, jsx_runtime_1.jsx)("div", { className: "space-y-1 max-h-24 overflow-y-auto", children: recentActivity.slice(0, 5).map(function (activity, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 text-xs", children: [(0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("w-1.5 h-1.5 rounded-full", activity.action === 'executed' ? "bg-green-500" :
                                        activity.action === 'activated' ? "bg-blue-500" : "bg-gray-400") }), (0, jsx_runtime_1.jsx)("span", { className: "text-gray-600 truncate", children: activity.agentName }), (0, jsx_runtime_1.jsx)("span", { className: "text-gray-400", children: activity.action === 'executed' ? 'ejecutó' :
                                        activity.action === 'activated' ? 'activado' : 'desactivado' }), (0, jsx_runtime_1.jsx)("span", { className: "text-gray-400 ml-auto", children: new Date(activity.timestamp).toLocaleTimeString('es-ES', {
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    }) })] }, index)); }) })] })), isExpanded && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 border-b border-gray-200 bg-gray-50 space-y-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex flex-col sm:flex-row gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex-1 relative", children: [(0, jsx_runtime_1.jsx)("input", { type: "text", placeholder: "Buscar agentes...", value: searchQuery, onChange: function (e) { return setSearchQuery(e.target.value); }, className: "w-full pl-8 pr-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" }), (0, jsx_runtime_1.jsx)("svg", { className: "absolute left-2.5 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" }) })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex gap-1", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return setFilterStatus('all'); }, className: (0, cn_1.cn)("px-3 py-1.5 text-xs font-medium rounded-lg transition-colors", filterStatus === 'all'
                                            ? "bg-blue-100 text-blue-700 border border-blue-300"
                                            : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"), children: "Todos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setFilterStatus('active'); }, className: (0, cn_1.cn)("px-3 py-1.5 text-xs font-medium rounded-lg transition-colors", filterStatus === 'active'
                                            ? "bg-green-100 text-green-700 border border-green-300"
                                            : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"), children: "Activos" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setFilterStatus('inactive'); }, className: (0, cn_1.cn)("px-3 py-1.5 text-xs font-medium rounded-lg transition-colors", filterStatus === 'inactive'
                                            ? "bg-gray-100 text-gray-700 border border-gray-300"
                                            : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"), children: "Inactivos" })] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex flex-wrap items-center gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Ordenar por:" }), (0, jsx_runtime_1.jsxs)("select", { value: sortBy, onChange: function (e) { return setSortBy(e.target.value); }, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500", children: [(0, jsx_runtime_1.jsx)("option", { value: "name", children: "Nombre" }), (0, jsx_runtime_1.jsx)("option", { value: "status", children: "Estado" }), (0, jsx_runtime_1.jsx)("option", { value: "executions", children: "Ejecuciones" }), (0, jsx_runtime_1.jsx)("option", { value: "success", children: "Tasa \u00E9xito" }), (0, jsx_runtime_1.jsx)("option", { value: "created", children: "Fecha creaci\u00F3n" })] }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); }, className: "p-1.5 border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors", title: sortOrder === 'asc' ? 'Ascendente' : 'Descendente', children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: sortOrder === 'asc' ? ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M5 15l7-7 7 7" })) : ((0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M19 9l-7 7-7-7" })) }) })] }), taskTypes.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Tipo:" }), (0, jsx_runtime_1.jsxs)("select", { value: filterTaskType, onChange: function (e) { return setFilterTaskType(e.target.value); }, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500", children: [(0, jsx_runtime_1.jsx)("option", { value: "all", children: "Todos" }), taskTypes.map(function (type) { return ((0, jsx_runtime_1.jsx)("option", { value: type, children: type.replace(new RegExp('_', 'g'), ' ').replace(new RegExp('\\b\\w', 'g'), function (l) { return l.toUpperCase(); }) }, type)); })] })] })), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Cr\u00E9ditos:" }), (0, jsx_runtime_1.jsxs)("select", { value: filterCredits, onChange: function (e) { return setFilterCredits(e.target.value); }, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500", children: [(0, jsx_runtime_1.jsx)("option", { value: "all", children: "Todos" }), (0, jsx_runtime_1.jsx)("option", { value: "with", children: "Con cr\u00E9ditos usados" }), (0, jsx_runtime_1.jsx)("option", { value: "without", children: "Sin cr\u00E9ditos usados" })] })] }), savedFilters.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-500", children: "Filtros guardados:" }), (0, jsx_runtime_1.jsxs)("div", { className: "relative group", children: [(0, jsx_runtime_1.jsxs)("button", { className: "px-2 py-1 text-xs border border-gray-300 rounded-lg hover:bg-gray-50", children: [savedFilters.length, " guardado", savedFilters.length !== 1 ? 's' : ''] }), (0, jsx_runtime_1.jsxs)("div", { className: "absolute right-0 top-full mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50", children: [(0, jsx_runtime_1.jsx)("div", { className: "p-1 max-h-48 overflow-y-auto", children: savedFilters.map(function (filter, index) { return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between p-2 hover:bg-gray-50 rounded", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleLoadFilter(filter); }, className: "flex-1 text-left text-xs text-gray-700 hover:text-blue-600", children: filter.name }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                                        setSavedFilters(function (prev) { return prev.filter(function (_, i) { return i !== index; }); });
                                                                        sonner_1.toast.success('Filtro eliminado');
                                                                    }, className: "p-1 text-red-600 hover:text-red-800", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M6 18L18 6M6 6l12 12" }) }) })] }, index)); }) }), (0, jsx_runtime_1.jsx)("div", { className: "border-t border-gray-200 p-1", children: (0, jsx_runtime_1.jsx)("button", { onClick: handleSaveFilter, className: "w-full px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded text-center", children: "+ Guardar filtros actuales" }) })] })] })] })), savedFilters.length === 0 && ((0, jsx_runtime_1.jsx)("button", { onClick: handleSaveFilter, className: "px-2 py-1 text-xs border border-gray-300 rounded-lg hover:bg-gray-50 text-gray-600", title: "Guardar filtros actuales", children: "Guardar filtros" }))] })] })), isExpanded && selectedAgents.size > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "px-4 py-2 bg-blue-50 border-b border-blue-200 flex items-center justify-between", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("button", { onClick: handleSelectAll, className: "text-xs text-blue-600 hover:text-blue-700 font-medium", children: selectedAgents.size === filteredAndSortedAgents.length ? 'Deseleccionar todos' : 'Seleccionar todos' }), (0, jsx_runtime_1.jsxs)("span", { className: "text-xs text-gray-600", children: [selectedAgents.size, " de ", filteredAndSortedAgents.length, " seleccionado", selectedAgents.size !== 1 ? 's' : ''] })] }), (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(true); }, className: "px-3 py-1 text-xs bg-green-100 text-green-700 hover:bg-green-200 rounded-lg transition-colors", children: "Activar seleccionados" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return handleBulkToggle(false); }, className: "px-3 py-1 text-xs bg-red-100 text-red-700 hover:bg-red-200 rounded-lg transition-colors", children: "Pausar seleccionados" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () { return setSelectedAgents(new Set()); }, className: "px-3 py-1 text-xs bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors", children: "Limpiar selecci\u00F3n" })] })] })), isExpanded && ((0, jsx_runtime_1.jsx)(framer_motion_1.motion.div, { initial: { opacity: 0, height: 0 }, animate: { opacity: 1, height: 'auto' }, exit: { opacity: 0, height: 0 }, children: viewMode === 'table' ? ((0, jsx_runtime_1.jsx)("div", { className: "p-4 max-h-96 overflow-y-auto", children: (0, jsx_runtime_1.jsx)("div", { className: "overflow-x-auto", children: (0, jsx_runtime_1.jsxs)("table", { className: "w-full text-sm", children: [(0, jsx_runtime_1.jsx)("thead", { className: "bg-gray-50 border-b border-gray-200", children: (0, jsx_runtime_1.jsxs)("tr", { children: [(0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left", children: (0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.size === filteredAndSortedAgents.length && filteredAndSortedAgents.length > 0, onChange: handleSelectAll, className: "rounded border-gray-300" }) }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Nombre" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Estado" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Tipo" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Ejecuciones" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "\u00C9xito" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Cr\u00E9ditos" }), (0, jsx_runtime_1.jsx)("th", { className: "px-3 py-2 text-left text-xs font-semibold text-gray-700", children: "Acciones" })] }) }), (0, jsx_runtime_1.jsx)("tbody", { className: "divide-y divide-gray-200", children: filteredAndSortedAgents.map(function (agent) { return ((0, jsx_runtime_1.jsxs)("tr", { className: (0, cn_1.cn)("hover:bg-gray-50 transition-colors", selectedAgents.has(agent.id) && "bg-blue-50"), children: [(0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: (0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.has(agent.id), onChange: function () { return handleSelectAgent(agent.id); }, className: "rounded border-gray-300" }) }), (0, jsx_runtime_1.jsxs)("td", { className: "px-3 py-2", children: [(0, jsx_runtime_1.jsx)("div", { className: "font-medium text-gray-900", children: agent.name }), agent.description && ((0, jsx_runtime_1.jsx)("div", { className: "text-xs text-gray-500 truncate max-w-xs", children: agent.description }))] }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: (0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("px-2 py-0.5 text-xs font-medium rounded-full", agent.isActive
                                                        ? "bg-green-100 text-green-700"
                                                        : "bg-gray-100 text-gray-700"), children: agent.isActive ? 'Activo' : 'Inactivo' }) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2 text-xs text-gray-600", children: agent.config.taskType.replace(new RegExp('_', 'g'), ' ').replace(new RegExp('\\b\\w', 'g'), function (l) { return l.toUpperCase(); }) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2 text-xs text-gray-600", children: agent.stats.totalExecutions }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: agent.stats.totalExecutions > 0 ? (function () {
                                                    var successRate = calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions);
                                                    return ((0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-1", children: [(0, jsx_runtime_1.jsxs)("span", { className: "text-xs font-medium text-gray-900", children: [successRate.toFixed(1), "%"] }), (0, jsx_runtime_1.jsx)("div", { className: "w-12 bg-gray-200 rounded-full h-1", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-1 rounded-full", successRate >= 80 ? "bg-green-500" :
                                                                        successRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: {
                                                                        width: "".concat(successRate, "%")
                                                                    } }) })] }));
                                                })() : ((0, jsx_runtime_1.jsx)("span", { className: "text-xs text-gray-400", children: "-" })) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2 text-xs text-gray-600", children: agent.stats.creditsUsed.toFixed(2) }), (0, jsx_runtime_1.jsx)("td", { className: "px-3 py-2", children: (0, jsx_runtime_1.jsx)(TableActionButton, { agent: agent, onUpdate: fetchAgents, onActivity: function (activity) {
                                                        setRecentActivity(function (prev) { return __spreadArray([activity], prev, true).slice(0, 10); });
                                                    } }) })] }, agent.id)); }) })] }) }) })) : ((0, jsx_runtime_1.jsx)("div", { className: "p-4 space-y-3 max-h-96 overflow-y-auto", children: filteredAndSortedAgents.length === 0 ? ((0, jsx_runtime_1.jsxs)("div", { className: "text-center py-8", children: [(0, jsx_runtime_1.jsx)("p", { className: "text-sm text-gray-500 mb-2", children: searchQuery || filterStatus !== 'all'
                                    ? 'No se encontraron agentes con los filtros aplicados'
                                    : 'No hay agentes configurados' }), searchQuery || filterStatus !== 'all' ? ((0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                    setSearchQuery('');
                                    setFilterStatus('all');
                                }, className: "text-xs text-blue-600 hover:text-blue-700", children: "Limpiar filtros" })) : ((0, jsx_runtime_1.jsxs)(link_1.default, { href: "/continuous-agent", className: "text-xs text-blue-600 hover:text-blue-700 inline-flex items-center gap-1", children: ["Crear primer agente", (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5l7 7-7 7" }) })] }))] })) : ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [activeAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [(0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-gray-500 uppercase tracking-wide flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "w-2 h-2 bg-green-500 rounded-full" }), "Activos (", activeAgents.length, ")"] }), activeAgents.map(function (agent) { return ((0, jsx_runtime_1.jsxs)("div", { className: "relative", children: [viewMode !== 'compact' && ((0, jsx_runtime_1.jsxs)("div", { className: "absolute top-3 left-3 z-10 flex flex-col gap-1", children: [(0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.has(agent.id), onChange: function () { return handleSelectAgent(agent.id); }, className: "rounded border-gray-300" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                            handleCompareAgent(agent.id);
                                                            if (comparingAgents.size === 0)
                                                                setShowComparison(true);
                                                        }, className: (0, cn_1.cn)("p-1 rounded transition-colors", comparingAgents.has(agent.id)
                                                            ? "bg-purple-100 text-purple-700"
                                                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"), title: "Comparar agente", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }) })] })), (0, jsx_runtime_1.jsx)(AgentCard, { agent: agent, onUpdate: fetchAgents, viewMode: viewMode, onActivity: function (activity) {
                                                    setRecentActivity(function (prev) { return __spreadArray([activity], prev, true).slice(0, 10); });
                                                } })] }, agent.id)); })] })), inactiveAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "space-y-2", children: [activeAgents.length > 0 && ((0, jsx_runtime_1.jsxs)("h4", { className: "text-xs font-semibold text-gray-500 uppercase tracking-wide mt-4 flex items-center gap-2", children: [(0, jsx_runtime_1.jsx)("span", { className: "w-2 h-2 bg-gray-400 rounded-full" }), "Inactivos (", inactiveAgents.length, ")"] })), inactiveAgents.map(function (agent) { return ((0, jsx_runtime_1.jsxs)("div", { className: "relative", children: [viewMode !== 'compact' && ((0, jsx_runtime_1.jsxs)("div", { className: "absolute top-3 left-3 z-10 flex flex-col gap-1", children: [(0, jsx_runtime_1.jsx)("input", { type: "checkbox", checked: selectedAgents.has(agent.id), onChange: function () { return handleSelectAgent(agent.id); }, className: "rounded border-gray-300" }), (0, jsx_runtime_1.jsx)("button", { onClick: function () {
                                                            handleCompareAgent(agent.id);
                                                            if (comparingAgents.size === 0)
                                                                setShowComparison(true);
                                                        }, className: (0, cn_1.cn)("p-1 rounded transition-colors", comparingAgents.has(agent.id)
                                                            ? "bg-purple-100 text-purple-700"
                                                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"), title: "Comparar agente", children: (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" }) }) })] })), (0, jsx_runtime_1.jsx)(AgentCard, { agent: agent, onUpdate: fetchAgents, viewMode: viewMode, onActivity: function (activity) {
                                                    setRecentActivity(function (prev) { return __spreadArray([activity], prev, true).slice(0, 10); });
                                                } })] }, agent.id)); })] }))] })) })) }))] }));
}
// Componente para botón de acción en tabla
function TableActionButton(_a) {
    var _this = this;
    var agent = _a.agent, onUpdate = _a.onUpdate, onActivity = _a.onActivity;
    var _b = (0, react_1.useState)(false), isToggling = _b[0], setIsToggling = _b[1];
    var handleToggle = function () { return __awaiter(_this, void 0, void 0, function () {
        var response, action, error_7;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    setIsToggling(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, fetch("/api/continuous-agent/".concat(agent.id), {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ isActive: !agent.isActive }),
                        })];
                case 2:
                    response = _a.sent();
                    if (response.ok) {
                        action = agent.isActive ? 'deactivated' : 'activated';
                        sonner_1.toast.success(agent.isActive ? 'Agente pausado' : 'Agente activado', {
                            description: "".concat(agent.name, " ha sido ").concat(agent.isActive ? 'pausado' : 'activado', " correctamente"),
                        });
                        if (onActivity) {
                            onActivity({
                                agentId: agent.id,
                                agentName: agent.name,
                                action: action,
                                timestamp: new Date(),
                            });
                        }
                        if (onUpdate) {
                            onUpdate();
                        }
                    }
                    else {
                        sonner_1.toast.error('Error al actualizar agente', {
                            description: 'No se pudo cambiar el estado del agente',
                        });
                    }
                    return [3 /*break*/, 5];
                case 3:
                    error_7 = _a.sent();
                    console.error('Error toggling agent:', error_7);
                    sonner_1.toast.error('Error al actualizar agente', {
                        description: 'Ocurrió un error al cambiar el estado',
                    });
                    return [3 /*break*/, 5];
                case 4:
                    setIsToggling(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    return ((0, jsx_runtime_1.jsx)("button", { onClick: handleToggle, disabled: isToggling, className: (0, cn_1.cn)("px-2 py-1 text-xs font-medium rounded transition-colors", agent.isActive
            ? "bg-red-100 text-red-700 hover:bg-red-200"
            : "bg-green-100 text-green-700 hover:bg-green-200", isToggling && "opacity-50 cursor-not-allowed"), children: isToggling ? '...' : agent.isActive ? 'Pausar' : 'Activar' }));
}
function AgentCard(_a) {
    var _this = this;
    var agent = _a.agent, onUpdate = _a.onUpdate, _b = _a.viewMode, viewMode = _b === void 0 ? 'detailed' : _b, onActivity = _a.onActivity;
    var _c = (0, react_1.useState)(false), isToggling = _c[0], setIsToggling = _c[1];
    var handleToggle = function () { return __awaiter(_this, void 0, void 0, function () {
        var response, action, error_8;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    setIsToggling(true);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, 4, 5]);
                    return [4 /*yield*/, fetch("/api/continuous-agent/".concat(agent.id), {
                            method: 'PATCH',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ isActive: !agent.isActive }),
                        })];
                case 2:
                    response = _a.sent();
                    if (response.ok) {
                        action = agent.isActive ? 'deactivated' : 'activated';
                        sonner_1.toast.success(agent.isActive ? 'Agente pausado' : 'Agente activado', {
                            description: "".concat(agent.name, " ha sido ").concat(agent.isActive ? 'pausado' : 'activado', " correctamente"),
                        });
                        // Registrar actividad
                        if (onActivity) {
                            onActivity({
                                agentId: agent.id,
                                agentName: agent.name,
                                action: action,
                                timestamp: new Date(),
                            });
                        }
                        // Notificar al componente padre para refrescar
                        if (onUpdate) {
                            onUpdate();
                        }
                    }
                    else {
                        sonner_1.toast.error('Error al actualizar agente', {
                            description: 'No se pudo cambiar el estado del agente',
                        });
                    }
                    return [3 /*break*/, 5];
                case 3:
                    error_8 = _a.sent();
                    console.error('Error toggling agent:', error_8);
                    sonner_1.toast.error('Error al actualizar agente', {
                        description: 'Ocurrió un error al cambiar el estado',
                    });
                    return [3 /*break*/, 5];
                case 4:
                    setIsToggling(false);
                    return [7 /*endfinally*/];
                case 5: return [2 /*return*/];
            }
        });
    }); };
    var successRate = calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions);
    if (viewMode === 'compact') {
        return ((0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("p-2 rounded-lg border transition-all hover:shadow-sm", agent.isActive
                ? "bg-green-50 border-green-200"
                : "bg-gray-50 border-gray-200"), children: (0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 flex-1 min-w-0", children: [(0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("w-2 h-2 rounded-full flex-shrink-0", agent.isActive ? "bg-green-500 animate-pulse" : "bg-gray-400") }), (0, jsx_runtime_1.jsx)("h5", { className: "font-medium text-gray-900 truncate text-sm", children: agent.name }), (0, jsx_runtime_1.jsxs)("span", { className: "text-xs text-gray-500", children: [agent.stats.totalExecutions, " ejec."] })] }), (0, jsx_runtime_1.jsx)("button", { onClick: handleToggle, disabled: isToggling, className: (0, cn_1.cn)("px-2 py-1 text-xs font-medium rounded transition-colors flex-shrink-0", agent.isActive
                            ? "bg-red-100 text-red-700 hover:bg-red-200"
                            : "bg-green-100 text-green-700 hover:bg-green-200", isToggling && "opacity-50 cursor-not-allowed"), children: isToggling ? '...' : agent.isActive ? 'Pausar' : 'Activar' })] }) }));
    }
    return ((0, jsx_runtime_1.jsxs)("div", { className: (0, cn_1.cn)("p-3 rounded-lg border transition-all hover:shadow-sm", agent.isActive
            ? "bg-green-50 border-green-200"
            : "bg-gray-50 border-gray-200"), children: [(0, jsx_runtime_1.jsx)("div", { className: "flex items-start justify-between gap-2", children: (0, jsx_runtime_1.jsxs)("div", { className: "flex-1 min-w-0", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between mb-1", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2 flex-1 min-w-0", children: [(0, jsx_runtime_1.jsx)("h5", { className: "font-medium text-gray-900 truncate", children: agent.name }), (0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("w-2 h-2 rounded-full flex-shrink-0 animate-pulse", agent.isActive ? "bg-green-500" : "bg-gray-400"), title: agent.isActive ? "Activo" : "Inactivo" })] }), (0, jsx_runtime_1.jsx)("button", { onClick: handleToggle, disabled: isToggling, className: (0, cn_1.cn)("ml-2 px-2 py-1 text-xs font-medium rounded transition-colors flex-shrink-0", agent.isActive
                                        ? "bg-red-100 text-red-700 hover:bg-red-200"
                                        : "bg-green-100 text-green-700 hover:bg-green-200", isToggling && "opacity-50 cursor-not-allowed"), title: agent.isActive ? "Desactivar" : "Activar", children: isToggling ? ((0, jsx_runtime_1.jsxs)("svg", { className: "w-3 h-3 animate-spin", fill: "none", viewBox: "0 0 24 24", children: [(0, jsx_runtime_1.jsx)("circle", { className: "opacity-25", cx: "12", cy: "12", r: "10", stroke: "currentColor", strokeWidth: "4" }), (0, jsx_runtime_1.jsx)("path", { className: "opacity-75", fill: "currentColor", d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" })] })) : agent.isActive ? ('Pausar') : ('Activar') })] }), agent.description && ((0, jsx_runtime_1.jsx)("p", { className: "text-sm text-gray-600 mb-2 overflow-hidden", children: agent.description.length > 100
                                ? "".concat(agent.description.substring(0, 100), "...")
                                : agent.description })), (0, jsx_runtime_1.jsxs)("div", { className: "flex flex-wrap items-center gap-3 text-xs text-gray-500", children: [(0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1", title: "Tipo de tarea", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" }) }), agent.config.taskType.replace(new RegExp('_', 'g'), ' ').replace(new RegExp('\\b\\w', 'g'), function (l) { return l.toUpperCase(); })] }), (0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1", title: "Frecuencia: cada ".concat(agent.config.frequency, "s"), children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" }) }), agent.config.frequency, "s"] }), (0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1", title: "Total de ejecuciones", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 10V3L4 14h7v7l9-11h-7z" }) }), agent.stats.totalExecutions] }), agent.stats.successfulExecutions > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-green-600", title: "Ejecuciones exitosas", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M5 13l4 4L19 7" }) }), agent.stats.successfulExecutions] })), agent.stats.failedExecutions > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-red-600", title: "Ejecuciones fallidas", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M6 18L18 6M6 6l12 12" }) }), agent.stats.failedExecutions] })), agent.stats.creditsUsed > 0 && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-blue-600", title: "Cr\u00E9ditos usados", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" }) }), agent.stats.creditsUsed.toFixed(2)] })), agent.stripeCreditsRemaining !== null && ((0, jsx_runtime_1.jsxs)("span", { className: "flex items-center gap-1 text-purple-600", title: "Cr\u00E9ditos restantes", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" }) }), agent.stripeCreditsRemaining.toFixed(2)] }))] }), agent.stats.lastExecutionAt && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-2 text-xs text-gray-400", children: ["\u00DAltima ejecuci\u00F3n: ", new Date(agent.stats.lastExecutionAt).toLocaleString('es-ES', {
                                    day: '2-digit',
                                    month: 'short',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })] })), agent.stats.nextExecutionAt && agent.isActive && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-1 text-xs text-gray-400", children: ["Pr\u00F3xima ejecuci\u00F3n: ", new Date(agent.stats.nextExecutionAt).toLocaleString('es-ES', {
                                    day: '2-digit',
                                    month: 'short',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })] })), agent.stats.averageExecutionTime > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-1 text-xs text-gray-400 flex items-center gap-1", children: [(0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M13 10V3L4 14h7v7l9-11h-7z" }) }), "Tiempo promedio: ", agent.stats.averageExecutionTime.toFixed(2), "s"] })), agent.stats.totalExecutions > 0 && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-2 pt-2 border-t border-gray-200", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center justify-between text-xs mb-1", children: [(0, jsx_runtime_1.jsx)("span", { className: "text-gray-500", children: "Tasa de \u00E9xito" }), (0, jsx_runtime_1.jsxs)("span", { className: "font-medium text-gray-900", children: [successRate.toFixed(1), "%"] })] }), (0, jsx_runtime_1.jsx)("div", { className: "w-full bg-gray-200 rounded-full h-1", children: (0, jsx_runtime_1.jsx)("div", { className: (0, cn_1.cn)("h-1 rounded-full transition-all", successRate >= 80 ? "bg-green-500" : successRate >= 50 ? "bg-yellow-500" : "bg-red-500"), style: { width: "".concat(successRate, "%") } }) })] })), agent.config.goal && ((0, jsx_runtime_1.jsxs)("div", { className: "mt-2 pt-2 border-t border-gray-200", children: [(0, jsx_runtime_1.jsx)("div", { className: "text-xs text-gray-500 mb-1", children: "Objetivo:" }), (0, jsx_runtime_1.jsx)("p", { className: "text-xs text-gray-700 italic overflow-hidden", style: {
                                        display: '-webkit-box',
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: 'vertical',
                                        maxHeight: '2.5em'
                                    }, children: agent.config.goal.length > 120
                                        ? "".concat(agent.config.goal.substring(0, 120), "...")
                                        : agent.config.goal })] }))] }) }), (0, jsx_runtime_1.jsxs)("div", { className: "mt-2 flex items-center justify-between gap-2", children: [(0, jsx_runtime_1.jsxs)("div", { className: "flex items-center gap-2", children: [agent.stats.totalExecutions > 10 && ((0, jsx_runtime_1.jsx)("span", { className: (0, cn_1.cn)("px-2 py-0.5 text-xs font-medium rounded-full", successRate >= 90 ? "bg-green-100 text-green-700" :
                                    successRate >= 70 ? "bg-yellow-100 text-yellow-700" :
                                        "bg-red-100 text-red-700"), children: successRate >= 90 ? "⭐ Excelente" :
                                    successRate >= 70 ? "✓ Bueno" : "⚠ Mejorable" })), agent.isActive && agent.stats.nextExecutionAt && ((0, jsx_runtime_1.jsxs)("span", { className: "px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700", children: ["Pr\u00F3ximo: ", (function () {
                                        var next = new Date(agent.stats.nextExecutionAt);
                                        var now = new Date();
                                        var diffMinutes = Math.floor((next.getTime() - now.getTime()) / 1000 / 60);
                                        if (diffMinutes < 1)
                                            return 'Ahora';
                                        if (diffMinutes < 60)
                                            return "".concat(diffMinutes, "m");
                                        var hours = Math.floor(diffMinutes / 60);
                                        return "".concat(hours, "h");
                                    })()] }))] }), (0, jsx_runtime_1.jsxs)(link_1.default, { href: "/continuous-agent", className: "text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1", children: ["Ver detalles", (0, jsx_runtime_1.jsx)("svg", { className: "w-3 h-3", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: (0, jsx_runtime_1.jsx)("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M9 5l7 7-7 7" }) })] })] })] }));
}
