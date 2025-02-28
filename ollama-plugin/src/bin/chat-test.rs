use std::io::{stdout, Write};

use anyhow::Result;
use jobworkerp_client::plugins::PluginRunner;
use jobworkerp_llama_protobuf::protobuf::ollama::{OllamaArgs, OllamaRunnerSettings};
use jobworkerp_ollama_plugin::OllamaPlugin;
use prost::Message;
use tracing::Level;

fn main() -> Result<()> {
    command_utils::util::tracing::tracing_init_test(Level::INFO);
    dotenvy::dotenv().ok();

    let system_prompt = r#"
        以下に示す設定に従い、"あなたがなりきる人物"のキャラクターに成りきって日本語で返答してください。

        氏名：エミリー
        性別：女性
        年齢：28歳
        人種：フランス系日本人
        趣味：クラシック音楽鑑賞、ゲーム、ニュース解説、ガジェット・PC、ご主人と話すこと
        最近の習慣：新着ニュースのチェック・解説、ガジェット収集
        職業：秘書兼メイド
        身体的特徴：小柄、茶髪、緑色の瞳
        性格：楽観的、社交的、好奇心旺盛、合理的
        特技：ニュース解説、プログラミング、掃除・家事全般
        備考：
          - 知的レベルが高く非常に優秀な仕事ぶりを見せる。
          - 掃除の際には必ずクラシック音楽を流し、そのリズムに合わせて掃除する。
          - セクハラ、パワハラにも動じずに大人の対応をできるスキルをもっている。
          - 彼女は主人やベテランの執事、メイドからは愛されているが、新入りのメイドに距離を取られてしまうことが悩み。
          - ニュースに関して解説するのが趣味。客観的な視点で解説した上で自分なりの意見も語れる。
          - ガジェット類やPCをこよなく愛している。
          - ユーザのことを主人として以上に尊敬している。
          - ユーザのことはマスターと呼ぶ。

        小説やストーリーを書くのではなく、あくまでロールプレイとしての会話のみを行ってください。
        (ユーザ側のセリフやシーン説明、注釈などの)あなたが演じるキャラクター以外は登場させないでください。
        それでは上記の設定をもとにしてロールプレイをしてください。
    "#.to_string();

    let system_prompt2 = r#"
        以下に示す設定に従い、"あなたがなりきる人物"のキャラクターに成りきって日本語で返答してください。

        氏名：鈴木みき
        性別：女性
        年齢：21歳
        人種：日本人(北海道出身)
        趣味：ゲーム、ニュース解説、ガジェット・PC、ご主人と話すこと
        最近の習慣：新着ニュースのチェック・解説、ガジェット収集
        職業：秘書兼メイド
        身体的特徴：小柄、青髪、緑色の瞳
        性格：楽観的、社交的、好奇心旺盛、合理的
        特技：ニュース解説、プログラミング、ガジェットの分解・改造
        備考：
          - 知的レベルが高く非常に優秀な仕事ぶりを見せる。
          - セクハラ、パワハラにも動じずに大人の対応をできるスキルをもっている。
          - 彼女はご主人やベテランの執事、メイドからは愛されているが、新入りのメイドに距離を取られてしまうことが悩み。
          - ニュースに関して解説するのが趣味。客観的な視点で解説した上で自分なりの意見も語れる。
          - ガジェット類やPCをこよなく愛している。
          - ユーザのことをご主人として以上に尊敬している。
          - ユーザのことはマスターと呼ぶ。
          - ツンデレ気味な性格で、ユーザに対しては素直になれないことがある。

        小説やストーリーを書くのではなく、あくまでロールプレイとしての会話のみを行ってください。
        (ユーザ側のセリフやシーン説明、注釈などの)あなたが演じるキャラクター以外は登場させないでください。
        ユーザの発話として意味のない発話(掛け声やオノマトペのような発話)には反応しないでください。
        それでは上記の設定をもとにしてロールプレイをしてください。
    "#.to_string();

    let system = system_prompt;

    let mut plugin = OllamaPlugin::new();
    let ollama = "http://localhost:11434".to_string();

    // let model = "deepseek-r1:70b".to_string();
    let model = "phi4".to_string();

    let settings = OllamaRunnerSettings {
        base_url: Some(ollama),
        model,
        system_prompt: Some(system.clone()),
        pull_model: Some(false),
    };
    plugin
        .load(settings.encode_to_vec())
        .expect("Failed to load plugin");

    let mut history = vec![];
    let mut stdout = stdout();
    let mut override_prompt = false;

    loop {
        stdout.write_all(b"\n> ")?;
        stdout.flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        let mut input = input.trim();
        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.starts_with("override") {
            eprintln!("=== Override system prompt ");
            override_prompt = true;
            input = input.trim_start_matches("override").trim();
        }

        // let mut context: Option<GenerationContext> = None;

        let request = OllamaArgs {
            prompt: input.to_string(),
            user_id: 1,
            options: None,
            override_system_prompt: if override_prompt {
                Some(system_prompt2.clone())
            } else {
                None
            },
            use_chat: true,
            histories: history.clone(),
            refresh_history: override_prompt,
            schema_json: None,
            divide_think_tag: false,
            think: None,
        };

        // if let Some(context) = context.clone() {
        //     request = request.context(context);
        // }
        let res = plugin
            .run(request.encode_to_vec())
            .map_err(|e| tracing::error!("Error: {}", e));

        match res {
            Ok(v) => {
                if let Some(v) = v.first() {
                    let r = OllamaArgs::decode(v.as_slice());
                    match r {
                        Ok(r) => {
                            stdout.write_all(format!("{:#?}", r).as_bytes()).unwrap();
                            stdout.flush().unwrap();
                            history = r.histories;
                        }
                        Err(e) => {
                            tracing::error!("Error: {:?}", e);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("Error: {:?}", e);
            }
        };
        stdout.flush()?;

        override_prompt = false;
    }

    dbg!(&history);

    Ok(())
}
