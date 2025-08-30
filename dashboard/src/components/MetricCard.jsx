export default function MetricCard({ label, value }) {
  return (
    <div className="card">
      <div className="sub">{label}</div>
      <div className="text-3xl font-bold mt-1">{value}</div>
    </div>
  );
}
