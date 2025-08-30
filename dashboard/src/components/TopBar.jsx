export default function TopBar({ lastRefreshed, onRefresh }) {
  return (
    <div className="flex items-center justify-between mb-5">
      <div>
        <div className="h1">Signals Dashboard</div>
        <div className="sub">Confirmed / Anticipative / Event-driven</div>
      </div>
      <div className="flex items-center gap-3">
        <span className="sub">Last refresh: {lastRefreshed}</span>
        <button onClick={onRefresh} className="px-3 py-2 rounded-xl bg-black text-white hover:opacity-90">
          Refresh
        </button>
      </div>
    </div>
  );
}
