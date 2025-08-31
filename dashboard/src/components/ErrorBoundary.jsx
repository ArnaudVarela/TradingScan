import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    // utile en debug
    console.error("[ErrorBoundary]", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 my-4 rounded border border-red-200 bg-red-50 text-sm text-red-800">
          <div className="font-semibold mb-1">Un module a rencontré une erreur.</div>
          <div className="opacity-80">
            {this.props.fallback || "Le reste de l’application reste utilisable."}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
