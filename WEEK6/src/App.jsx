import { useState, useEffect, useRef } from "react";

const FONT_LINK = document.createElement("link");
FONT_LINK.rel = "stylesheet";
FONT_LINK.href = "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap";
document.head.appendChild(FONT_LINK);

const SEV_COLOR = { LOW:"#39ff14", MEDIUM:"#ffd700", HIGH:"#ff6b00", CRITICAL:"#ff003c" };
const SEV_BG    = { LOW:"#0a1f0a", MEDIUM:"#1f1800", HIGH:"#1f0e00", CRITICAL:"#1f0008" };

function getSeverity(conf) {
  if (conf > 0.9)  return "CRITICAL";
  if (conf > 0.75) return "HIGH";
  if (conf > 0.6)  return "MEDIUM";
  return "LOW";
}

function Sparkline({ data, color = "#00ff88", height = 70 }) {
  const ref = useRef();
  useEffect(() => {
    const c = ref.current;
    if (!c || data.length < 2) return;
    const ctx = c.getContext("2d");
    c.width = c.offsetWidth;
    c.height = height;
    ctx.clearRect(0, 0, c.width, c.height);
    const max = Math.max(...data, 1);
    const w = c.width / (data.length - 1);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    data.forEach((v, i) => {
      const x = i * w, y = c.height - (v / max) * (c.height - 4) - 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.beginPath();
    ctx.fillStyle = color + "18";
    data.forEach((v, i) => {
      const x = i * w, y = c.height - (v / max) * (c.height - 4) - 2;
      if (i === 0) ctx.moveTo(x, c.height);
      ctx.lineTo(x, y);
    });
    ctx.lineTo((data.length - 1) * w, c.height);
    ctx.fill();
  }, [data]);
  return <canvas ref={ref} style={{ width: "100%", display: "block", height }} />;
}

export default function App() {
  const [alerts, setAlerts]           = useState([]);
  const [traffic, setTraffic]         = useState([]);
  const [threatTraffic, setThreat]    = useState([]);
  const [paused, setPaused]           = useState(false);
  const [connected, setConnected]     = useState(true);
  const [clock, setClock]             = useState("");
  const [uptime, setUptime]           = useState(0);
  const [metrics, setMetrics]         = useState({ total:0, threats:0, critical:0, ips:0 });
  const [threatCounts, setThreatCounts] = useState({});
  const [hexActive, setHexActive]     = useState(0);

  const pausedRef     = useRef(false);
  const alertsRef     = useRef([]);
  const metricsRef    = useRef({ total:0, threats:0, critical:0 });
  const uniqueIPs     = useRef(new Set());
  const threatCountsRef = useRef({});

  useEffect(() => {
    const interval = setInterval(async () => {
      if (pausedRef.current) return;
      try {
        const res  = await fetch("http://localhost:8000/alerts");
        const data = await res.json();
        setConnected(true);

        const m = metricsRef.current;
        m.total += data.length;
        const threats = data.filter(a => a.label?.toLowerCase() !== "benign");
        m.threats += threats.length;

        data.forEach(a => {
          uniqueIPs.current.add(a.src_ip);
          if (getSeverity(a.confidence) === "CRITICAL") m.critical++;
          if (a.label?.toLowerCase() !== "benign") {
            threatCountsRef.current[a.label] = (threatCountsRef.current[a.label] || 0) + 1;
          }
        });

        if (data.length > 0) {
          const now = new Date().toLocaleTimeString();
          const fresh = data.map(a => ({ ...a, time: now }));
          alertsRef.current = [...fresh, ...alertsRef.current].slice(0, 100);
          setAlerts([...alertsRef.current]);
        }

        setTraffic(t => [...t, data.length].slice(-60));
        setThreat(t => [...t, threats.length].slice(-60));
        setMetrics({ total: m.total, threats: m.threats, critical: m.critical, ips: uniqueIPs.current.size });
        setThreatCounts({ ...threatCountsRef.current });
        setHexActive(data.length);
      } catch {
        setConnected(false);
        setTraffic(t => [...t, 0].slice(-60));
        setThreat(t => [...t, 0].slice(-60));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const t = setInterval(() => {
      setClock(new Date().toLocaleTimeString());
      setUptime(u => u + 1);
    }, 1000);
    return () => clearInterval(t);
  }, []);

  const togglePause = () => { pausedRef.current = !pausedRef.current; setPaused(p => !p); };
  const clearAll    = () => { alertsRef.current = []; setAlerts([]); };

  const fmtUptime = s => {
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
    return h ? `${h}h ${m}m ${sec}s` : m ? `${m}m ${sec}s` : `${sec}s`;
  };

  const topThreats = Object.entries(threatCounts).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const maxThreat  = topThreats[0]?.[1] || 1;

  const S = {
    root: { background:"#020c06", color:"#00ff88", fontFamily:"'Share Tech Mono',monospace", fontSize:13, padding:16, minHeight:"100vh" },
    panel: { background:"#0d2218", border:"1px solid #00ff8833", borderRadius:6, padding:14, marginBottom:12 },
    panelTitle: { fontFamily:"'Orbitron',monospace", fontSize:10, letterSpacing:3, color:"#00ff8880", marginBottom:12, textTransform:"uppercase", display:"flex", justifyContent:"space-between", alignItems:"center" },
    th: { fontSize:10, letterSpacing:2, color:"#00ff8866", padding:"8px 10px", textAlign:"left", borderBottom:"1px solid #00ff8833", textTransform:"uppercase", fontWeight:400 },
    td: { padding:"7px 10px", borderBottom:"1px solid #00ff8811", fontSize:12, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" },
  };

  return (
    <div style={S.root}>
      {/* Scanline */}
      <style>{`
        @keyframes scan{0%{top:0}100%{top:100vh}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        @keyframes flashin{0%{background:#00ff8822}100%{background:transparent}}
        .sev-CRITICAL{animation:pulse 1s infinite}
        .new-row td{animation:flashin .4s ease}
        tr:hover td{background:#00ff8808}
      `}</style>
      <div style={{position:"fixed",top:0,left:0,right:0,height:2,background:"linear-gradient(90deg,transparent,#00ff8844,transparent)",animation:"scan 4s linear infinite",pointerEvents:"none",zIndex:999}} />

      {/* Header */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:18,borderBottom:"1px solid #00ff8833",paddingBottom:14}}>
        <div>
          <h1 style={{fontFamily:"'Orbitron',monospace",fontSize:22,fontWeight:900,letterSpacing:4,color:"#00ff88",margin:0}}>NETGUARD IDS</h1>
          <div style={{display:"flex",alignItems:"center",gap:8,marginTop:6,fontSize:11}}>
            <div style={{width:8,height:8,borderRadius:"50%",background:"#00ff88",animation:"pulse 1.5s infinite"}} />
            <span style={{color:"#00ff8870",letterSpacing:2}}>SYSTEM OPERATIONAL</span>
            <span style={{color:"#00ff8840",margin:"0 4px"}}>|</span>
            <span style={{color:"#00ff8870"}}>{clock}</span>
            <span style={{color:"#00ff8840",margin:"0 4px"}}>|</span>
            <div style={{width:6,height:6,borderRadius:"50%",background:connected?"#00ff88":"#ff003c",animation:connected?"pulse 1.5s infinite":"none"}} />
            <span style={{color:connected?"#00ff88":"#ff003c",letterSpacing:1}}>{connected?"BACKEND CONNECTED":"BACKEND OFFLINE"}</span>
          </div>
        </div>
        <div style={{display:"flex",gap:8}}>
          <button onClick={togglePause} style={{background:"transparent",border:`1px solid ${paused?"#ff6b0088":"#00ff8866"}`,color:paused?"#ff6b00":"#00ff88",fontFamily:"'Share Tech Mono',monospace",fontSize:11,padding:"5px 12px",cursor:"pointer",letterSpacing:1,borderRadius:3}}>
            {paused ? "RESUME" : "PAUSE"}
          </button>
          <button onClick={clearAll} style={{background:"transparent",border:"1px solid #00ff8866",color:"#00ff88",fontFamily:"'Share Tech Mono',monospace",fontSize:11,padding:"5px 12px",cursor:"pointer",letterSpacing:1,borderRadius:3}}>
            CLEAR
          </button>
        </div>
      </div>

      {/* Metrics */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:12}}>
        {[
          {label:"Total Flows",  val:metrics.total.toLocaleString(),   sub:"packets captured",  color:"#00ff88"},
          {label:"Threats",      val:metrics.threats.toLocaleString(),  sub:"malicious detected",color:"#ff003c"},
          {label:"Critical",     val:metrics.critical.toLocaleString(), sub:"high severity",     color:"#ffd700"},
          {label:"Unique IPs",   val:metrics.ips.toLocaleString(),      sub:"sources tracked",   color:"#00cfff"},
        ].map(m => (
          <div key={m.label} style={{background:"#0d2218",border:"1px solid #00ff8833",borderRadius:6,padding:"12px 14px",borderTop:`2px solid ${m.color}`}}>
            <div style={{fontSize:10,letterSpacing:2,color:"#00ff8870",marginBottom:6,textTransform:"uppercase"}}>{m.label}</div>
            <div style={{fontFamily:"'Orbitron',monospace",fontSize:24,fontWeight:700,color:m.color}}>{m.val}</div>
            <div style={{fontSize:10,color:"#00ff8850",marginTop:4}}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* Charts + Threat List */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 280px",gap:12,marginBottom:12}}>
        <div style={S.panel}>
          <div style={S.panelTitle}>
            <span>Traffic Telemetry</span>
            <span style={{fontSize:9,color:"#00ff88",border:"1px solid #00ff88",padding:"2px 6px",borderRadius:2,animation:"pulse 2s infinite"}}>LIVE</span>
          </div>
          <Sparkline data={traffic} color="#00ff88" height={70} />
          <Sparkline data={threatTraffic} color="#ff003c" height={40} />
        </div>
        <div style={S.panel}>
          <div style={S.panelTitle}><span>Top Threat Types</span></div>
          <div style={{display:"flex",flexDirection:"column",gap:6,maxHeight:150,overflowY:"auto"}}>
            {topThreats.length === 0
              ? <div style={{color:"#00ff8830",fontSize:11,padding:8}}>No threats detected</div>
              : topThreats.map(([k,v]) => (
                <div key={k} style={{background:"#020c06",border:"1px solid #00ff8822",borderRadius:4,padding:"7px 10px",fontSize:11}}>
                  <div style={{display:"flex",justifyContent:"space-between"}}>
                    <span style={{color:"#ff6b00"}}>{k}</span>
                    <span style={{color:"#00ff8880"}}>{v}</span>
                  </div>
                  <div style={{marginTop:4,height:3,background:"#00ff8815",borderRadius:2}}>
                    <div style={{height:3,borderRadius:2,background:"#00ff88",width:`${Math.round(v/maxThreat*100)}%`,transition:"width .5s"}} />
                  </div>
                </div>
              ))
            }
          </div>
        </div>
      </div>

      {/* Alert Table */}
      <div style={S.panel}>
        <div style={S.panelTitle}>
          <span>Intrusion Alert Feed</span>
          <span style={{fontSize:10,color:"#00ff8860"}}>{alerts.length} alerts</span>
        </div>
        <table style={{width:"100%",borderCollapse:"collapse",tableLayout:"fixed"}}>
          <thead>
            <tr>
              {["Time","Source IP","Destination","Attack Type","Confidence","Severity"].map(h => (
                <th key={h} style={S.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {alerts.length === 0
              ? <tr><td colSpan={6} style={{...S.td,textAlign:"center",color:"#00ff8830",padding:20}}>Awaiting threat data...</td></tr>
              : alerts.slice(0,30).map((a, i) => {
                  const sev  = getSeverity(a.confidence);
                  const col  = SEV_COLOR[sev];
                  const pct  = Math.round(a.confidence * 100);
                  return (
                    <tr key={i} className={i===0?"new-row":""} style={{background:SEV_BG[sev]+"44"}}>
                      <td style={{...S.td,color:"#00ff8870"}}>{a.time}</td>
                      <td style={{...S.td,color:"#00cfff"}}>{a.src_ip}</td>
                      <td style={{...S.td,color:"#00cfff99"}}>{a.dst_ip}</td>
                      <td style={{...S.td,color:"#ff6b00"}}>{a.label}</td>
                      <td style={S.td}>
                        <span style={{color:col}}>{pct}%</span>
                        <div style={{height:3,borderRadius:2,background:col,width:`${pct}%`,marginTop:3}} />
                      </td>
                      <td style={S.td}>
                        <span className={`sev-${sev}`} style={{fontSize:10,padding:"2px 8px",borderRadius:2,letterSpacing:1,fontWeight:700,color:col,border:`1px solid ${col}66`,background:SEV_BG[sev]}}>
                          {sev}
                        </span>
                      </td>
                    </tr>
                  );
                })
            }
          </tbody>
        </table>
      </div>

      {/* Bottom panels */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
        <div style={S.panel}>
          <div style={S.panelTitle}><span>Activity Matrix</span></div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(8,1fr)",gap:3,marginTop:4}}>
            {Array.from({length:32},(_,i) => {
              const active = i < Math.min(hexActive, 32);
              const col = i < hexActive*0.3 ? "#ff003c" : i < hexActive*0.6 ? "#ff6b00" : "#00ff88";
              return <div key={i} style={{aspectRatio:1,borderRadius:3,background:active?col:"#00ff88",opacity:active?1:0.1,transition:"all .3s"}} />;
            })}
          </div>
        </div>
        <div style={S.panel}>
          <div style={S.panelTitle}><span>System Diagnostics</span></div>
          <div style={{display:"flex",flexDirection:"column",gap:8,fontSize:11}}>
            {[
              ["Model",     "XGBoost + LSTM Ensemble", "#00ff88"],
              ["Interface", "en0 (macOS)",              "#00cfff"],
              ["Poll Rate", "1000ms",                   "#00ff88"],
              ["Max Alerts","100",                      "#00ff88"],
              ["Backend",   "127.0.0.1:8000",           "#00ff88"],
              ["Uptime",    fmtUptime(uptime),           "#ffd700"],
            ].map(([k,v,c]) => (
              <div key={k} style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{color:"#00ff8870"}}>{k}</span>
                <span style={{color:c}}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#00ff8840",letterSpacing:1,marginTop:8}}>
        <span>NETGUARD IDS v2.0 // ENSEMBLE DETECTION ENGINE</span>
        <span>{clock}</span>
      </div>
    </div>
  );
}