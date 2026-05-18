// ============================================================================
// DASHBOARD DE DIAGNÓSTICO DE VÁLVULAS - VERSIÓN CORREGIDA
// ============================================================================

const API_URL = "http://localhost:5000/api";

let valvesData = [];
let selectedValveId = null;
let chartInstance = null;

// ============================================================================
// FUNCIONES API
// ============================================================================

async function cargarValvulas() {
    try {
        console.log("Cargando válvulas desde API...");
        const response = await fetch(`${API_URL}/valvulas`);
        if (!response.ok) throw new Error('Error en API: ' + response.status);
        valvesData = await response.json();
        console.log(`✅ ${valvesData.length} válvulas cargadas`);
        
        renderHeaderStats();
        renderValveList();
        
        return valvesData;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('valve-items').innerHTML = 
            '<div style="padding: 20px; text-align: center; color: #ff6b35;">❌ Error: No se pudo conectar con la API</div>';
        return [];
    }
}

async function cargarResumen() {
    try {
        const response = await fetch(`${API_URL}/resumen`);
        if (!response.ok) throw new Error('Error en API');
        return await response.json();
    } catch (error) {
        console.error("Error cargando resumen:", error);
        return null;
    }
}

async function cargarDetalleValvula(id) {
    try {
        const response = await fetch(`${API_URL}/valvulas/${id}`);
        if (!response.ok) throw new Error('Error en API');
        return await response.json();
    } catch (error) {
        console.error("Error cargando detalle:", error);
        return null;
    }
}

// ============================================================================
// RENDERIZADO
// ============================================================================

async function renderHeaderStats() {
    const resumen = await cargarResumen();
    if (resumen) {
        document.getElementById('header-stats').innerHTML = `
            <div class="stat-badge severo"><span class="dot"></span>${resumen.alerta || 0} ALERTA</div>
            <div class="stat-badge moderado"><span class="dot"></span>${resumen.aceptable_com || 0} ACEPTABLE CON COMENTARIOS</div>
            <div class="stat-badge normal"><span class="dot"></span>${resumen.aceptable || 0} ACEPTABLE</div>
            <div class="total-count"><strong>${resumen.total || 0}</strong> válvulas</div>
        `;
    }
}

function renderValveList() {
    if (!valvesData.length) return;
    
    const severos = valvesData.filter(v => v.status === "ALERTA").length;
    const moderados = valvesData.filter(v => v.status === "ACEPTABLE CON COMENTARIOS").length;
    const normales = valvesData.filter(v => v.status === "ACEPTABLE").length;
    
    document.getElementById('valve-count-sub').textContent = `${valvesData.length} válvulas diagnosticadas por ML`;
    document.getElementById('list-badges').innerHTML = `
        <span class="badge severo">${severos} ALERTA</span>
        <span class="badge moderado">${moderados} ACEPTABLE COM</span>
        <span class="badge normal">${normales} ACEPTABLE</span>
    `;
    
    const container = document.getElementById('valve-items');
    container.innerHTML = valvesData.map(valve => {
        let severidadClass = "normal";
        let severidadTexto = "Normal";
        if (valve.status === "ALERTA") {
            severidadClass = "severo";
            severidadTexto = "ALERTA";
        } else if (valve.status === "ACEPTABLE CON COMENTARIOS") {
            severidadClass = "moderado";
            severidadTexto = "ACEPTABLE CON COMENTARIOS";
        }
        
        // Extraer valores del array measurements
        const measurements = valve.measurements || [];
        const getMedida = (nombre) => {
            const item = measurements.find(m => m.name === nombre);
            return item ? { measured: item.measured, recommended: item.recommended } : { measured: 0, recommended: 0 };
        };
        
        const friccion = getMedida('Fricción');
        const carga = getMedida('Carga en Asiento/Torque');
        const banda = getMedida('Banda Muerta');
        const linealidad = getMedida('Linealidad Dinámica');
        
        const getNivel = (med, rec) => {
            const diff = Math.abs((med || 0) - (rec || 0));
            if (diff > 10) return "severo";
            if (diff > 5) return "moderado";
            return "normal";
        };
        
        const niveles = {
            friccion: getNivel(friccion.measured, friccion.recommended),
            carga: getNivel(carga.measured, carga.recommended),
            banda: getNivel(banda.measured, banda.recommended),
            linealidad: getNivel(linealidad.measured, linealidad.recommended)
        };
        
        const dots = `
            <div class="valve-dot-item"><span class="dot-indicator ${niveles.friccion}"></span><span class="dot-label">Fri</span></div>
            <div class="valve-dot-item"><span class="dot-indicator ${niveles.carga}"></span><span class="dot-label">Car</span></div>
            <div class="valve-dot-item"><span class="dot-indicator ${niveles.banda}"></span><span class="dot-label">Ban</span></div>
            <div class="valve-dot-item"><span class="dot-indicator ${niveles.linealidad}"></span><span class="dot-label">Lin</span></div>
        `;
        
        return `
            <button class="valve-item ${selectedValveId === valve.id ? 'selected' : ''}"
                    onclick="selectValve(${valve.id})">
                <div class="valve-item-top">
                    <div>
                        <div class="valve-name">${valve.name}</div>
                        <div class="valve-status">${valve.status}</div>
                    </div>
                    <span class="badge ${severidadClass}">${severidadTexto}</span>
                </div>
                <div class="valve-dots">${dots}</div>
            </button>
        `;
    }).join('');
}

async function renderDetail(valveId) {
    try {
        const valve = await cargarDetalleValvula(valveId);
        if (!valve) return;
        
        document.getElementById('empty-state').style.display = 'none';
        document.getElementById('valve-detail').style.display = 'block';
        
        let severidadClass = "normal", severidadTexto = "Normal";
        if (valve.status === "ALERTA") {
            severidadClass = "severo";
            severidadTexto = "ALERTA";
        } else if (valve.status === "ACEPTABLE CON COMENTARIOS") {
            severidadClass = "moderado";
            severidadTexto = "ACEPTABLE CON COMENTARIOS";
        }
        
        document.getElementById('detail-name').textContent = `Válvula ${valve.name}`;
        document.getElementById('detail-status').textContent = valve.status;
        document.getElementById('detail-severity-badge').innerHTML = `<span class="badge ${severidadClass}">${severidadTexto}</span>`;
        
        // Recomendación principal (usar la primera recomendación relevante)
        let recommendationText = valve.recommendation || "Revisar parámetros individuales";
        document.getElementById('detail-recommendation').textContent = recommendationText;
        
        // Plan de acción
        let actionText = '';
        if (valve.actionPlan) {
            if (typeof valve.actionPlan === 'object') {
                actionText = Object.values(valve.actionPlan).filter(v => v && v !== "Sin diagnóstico").join(' | ');
            } else {
                actionText = valve.actionPlan;
            }
        }
        document.getElementById('detail-action').textContent = actionText || "Monitoreo programado";
        
        renderChart(valve);
        renderParamCards(valve);
        
    } catch (error) {
        console.error("Error en detalle:", error);
    }
}

function renderChart(valve) {
    const canvas = document.getElementById('valveChart');
    if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
    
    const measurements = valve.measurements || [];
    
    const getValor = (nombre) => {
        const item = measurements.find(m => m.name === nombre);
        return item ? { measured: item.measured, recommended: item.recommended } : { measured: 0, recommended: 0 };
    };
    
    const friccion = getValor('Fricción');
    const carga = getValor('Carga en Asiento/Torque');
    const banda = getValor('Banda Muerta');
    const linealidad = getValor('Linealidad Dinámica');
    
    const labels = ['Fricción', 'Carga', 'Banda Muerta', 'Linealidad'];
    const measured = [friccion.measured, carga.measured, banda.measured, linealidad.measured];
    const recommended = [friccion.recommended, carga.recommended, banda.recommended, linealidad.recommended];
    
    chartInstance = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Valor Medido', data: measured, backgroundColor: 'rgba(6,182,212,0.75)', borderColor: '#06b6d4', borderWidth: 1, borderRadius: 4 },
                { label: 'Valor Recomendado', data: recommended, backgroundColor: 'rgba(34,197,94,0.75)', borderColor: '#22c55e', borderWidth: 1, borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#f1f5f9', usePointStyle: true, pointStyle: 'rect' } },
                tooltip: { backgroundColor: '#1c2333', borderColor: '#2a3145', borderWidth: 1, titleColor: '#f1f5f9', bodyColor: '#8b9bba' }
            },
            scales: {
                x: { ticks: { color: '#f1f5f9', font: { size: 11 } }, grid: { color: 'rgba(42,49,69,0.4)' } },
                y: { ticks: { color: '#f1f5f9' }, grid: { color: 'rgba(42,49,69,0.4)' }, title: { display: true, text: 'Unidades', color: '#f1f5f9' } }
            }
        }
    });
}

function renderParamCards(valve) {
    const measurements = valve.measurements || [];
    
    const getSeverity = (med, rec, umbralMod, umbralSev) => {
        const diff = Math.abs((med || 0) - (rec || 0));
        if (diff > umbralSev) return "severo";
        if (diff > umbralMod) return "moderado";
        return "normal";
    };
    
    const severidadMap = { 'normal': 'Normal', 'moderado': 'Moderado', 'severo': 'Severo' };
    
    const diagnosisMap = {
        'Fricción': { severo: 'Severa. Posible desgaste de packing.', moderado: 'Moderada. Requiere monitoreo.', normal: 'Normal. Dentro de especificaciones.' },
        'Carga en Asiento/Torque': { severo: 'Severa. Riesgo de falla.', moderado: 'Moderada. Considerar ajuste.', normal: 'Normal. Dentro de especificaciones.' },
        'Banda Muerta': { severo: 'Severa. Problemas graves en posicionador.', moderado: 'Moderada. Calibración de posicionador.', normal: 'Normal. Dentro de especificaciones.' },
        'Linealidad Dinámica': { severo: 'Severa. Problemas de control graves.', moderado: 'Moderada. Verificar caracterización.', normal: 'Normal. Dentro de especificaciones.' }
    };
    
    // Función para obtener el valor de un parámetro por nombre
    const getParam = (nombre) => {
        const item = measurements.find(m => m.name === nombre);
        return item ? { measured: item.measured, recommended: item.recommended, severity: item.severity } : { measured: 0, recommended: 0, severity: 'normal' };
    };
    
    const friccion = getParam('Fricción');
    const carga = getParam('Carga en Asiento/Torque');
    const banda = getParam('Banda Muerta');
    const linealidad = getParam('Linealidad Dinámica');
    
    const params = [
        { name: 'Fricción', measured: friccion.measured, recommended: friccion.recommended, severity: friccion.severity || getSeverity(friccion.measured, friccion.recommended, 5, 10) },
        { name: 'Carga en Asiento/Torque', measured: carga.measured, recommended: carga.recommended, severity: carga.severity || getSeverity(carga.measured, carga.recommended, 50, 100) },
        { name: 'Banda Muerta', measured: banda.measured, recommended: banda.recommended, severity: banda.severity || getSeverity(banda.measured, banda.recommended, 0.5, 1) },
        { name: 'Linealidad Dinámica', measured: linealidad.measured, recommended: linealidad.recommended, severity: linealidad.severity || getSeverity(linealidad.measured, linealidad.recommended, 0.5, 1) }
    ];
    
    document.getElementById('param-cards').innerHTML = params.map(p => {
        const diag = diagnosisMap[p.name]?.[p.severity] || 'Sin diagnóstico';
        return `
            <div class="param-card ${p.severity}">
                <div class="param-top">
                    <span class="param-name">${p.name}</span>
                    <span class="badge outline-${p.severity}">${severidadMap[p.severity]}</span>
                </div>
                <div class="param-diag">${diag}</div>
                <div class="param-values">
                    <div class="param-value-box">
                        <span class="param-value-label">Medido</span>
                        <span class="param-value-num value-measured">${p.measured.toFixed(2)}<span class="param-value-unit">unid</span></span>
                    </div>
                    <div class="param-value-box">
                        <span class="param-value-label">Recomendado</span>
                        <span class="param-value-num value-recommended">${p.recommended.toFixed(2)}<span class="param-value-unit">unid</span></span>
                    </div>
                    <div class="param-value-box">
                        <span class="param-value-label">Desviación</span>
                        <span class="param-value-num value-deviation-${p.severity}">${Math.abs(p.measured - p.recommended).toFixed(2)}<span class="param-value-unit">unid</span></span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

async function selectValve(id) {
    selectedValveId = id;
    renderValveList();
    await renderDetail(id);
}

// Inicializar
cargarValvulas();