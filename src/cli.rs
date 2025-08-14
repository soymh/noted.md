use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "A command-line tool to convert handwritten notes into clean and readable Markdown files",
    long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Convert files to Markdown format
    Convert {
        /// Path to a file or directory to convert
        #[arg(required = true)]
        path: String,

        /// Output directory to save converted files
        #[arg(
            short,
            long,
            help = "Directory where converted markdown files will be saved"
        )]
        output: Option<String>,

        /// API key for conversion
        #[arg(long, env = "GEMINI_API_KEY", hide_env_values = true)]
        api_key: Option<String>,

        // Prompt the LLM
        #[arg(short, long, help = "Add a custom prompt to pass to the LLM")]
        prompt: Option<String>,

        /// Model to use for conversion
        #[arg(long, help = "Model to use for conversion (e.g., gemini-pro-vision, claude-3-opus-20240229, llava)")]
        model: Option<String>,

        /// Number of pages to process at once (max 30)
        #[arg(long, default_value = "1", help = "Number of pages to process at once (max 30)")]
        pages_per_batch: u32,

        /// Specific pages or page ranges to convert (e.g., "1,3-5,8")
        #[arg(long, help = "Specific pages or page ranges to convert (e.g., \"1,3-5,8\")")]
        pages: Option<String>,
    },

    /// Configure notedmd settings
    Config {
        /// Set your Gemini API key
        #[arg(long, help = "Set your Gemini API key for future use")]
        set_api_key: Option<String>,

        /// Set your Claude API key
        #[arg(long, help = "Set your Claude API key for future use")]
        set_claude_api_key: Option<String>,

        // Set active provider
        #[arg(long, help = "Set the active provider")]
        set_provider: Option<String>,

        /// Show config file location
        #[arg(long, help = "Shows the location of your configuration file")]
        show_path: bool,

        // Show config file
        #[arg(long, help = "Shows the content of your configuration")]
        show: bool,

        // Trigger onboarding flow
        #[arg(long, help = "Edit the configuration file")]
        edit: bool,
    },
}

