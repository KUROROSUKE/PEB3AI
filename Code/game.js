let p1_hand = []; let p2_hand = []
let p1_point = 0; let p2_point = 0
let p1_selected_card = []; let p2_selected_card = []

const card_num = 8
let WIN_POINT = card_num*30+10
let WIN_TURN = 10

let dropped_cards_p1 = []; let dropped_cards_p2 = []

let turn = "p1"
let time = "game"
let numTurn = 1
let p1_is_acting = false

const elementToNumber = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,"Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,"Fe": 26, "Cu": 29, "Zn": 30, "I": 53}
const elements = [...Array(6).fill('H'), ...Array(4).fill('O'), ...Array(4).fill('C'),'He', 'Li', 'Be', 'B', 'N', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca','Fe', 'Cu', 'Zn', 'I']
const element = ['H','O','C','He', 'Li', 'Be', 'B', 'N', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca','Fe', 'Cu', 'Zn', 'I']
let deck = [...elements, ...elements]
let materials = []
let imageCache = {}

let model;
let modelName;

const countTemplate = Object.fromEntries(Object.values(elementToNumber).map(num => [num, 0]));


function convertToCount() {
    // テンプレートのコピーを作成
    let count = { ...countTemplate };
    // 配列内の各元素をカウント
    dropped_cards_p2.forEach(elem => {
        let num = elementToNumber[elem];
        if (num !== undefined) {
            count[num] += 1;
        }
    });
    // カウントの値を配列として返す（数値順に並ぶ）
    return Object.values(count);
}

let xs = [];
let ys = [];
let isTraining = false; // 学習中フラグ

function extractModelName(url) {
    const match = url.match(/\/([^\/]+)$/);
    return match ? match[1] : null;
}

// 1. モデルをロード（localStorageを優先）
async function loadModel(url=null, modelName=null) {
    try {
        if (url == null){//最初にこれを読み込む
            const models = await tf.io.listModels();
            modelName = "model1";
            if (models['indexeddb://model1']) {
                model = await tf.loadLayersModel('indexeddb://model1'); // IndexedDB からロード
                console.log("ローカルの学習済みモデルをロードしました");
            } else {
                model = await tf.loadLayersModel('https://kurorosuke.github.io/AI_models/model1/model.json'); // 外部モデルをロード
                console.log("サーバーからモデルをロードしました");
        }} else  {
            const models = await tf.io.listModels();
            modelName = modelName==null ? extractModelName(url) : modelName
            console.log(modelName);
            if (models[`indexeddb://${modelName}`]) {
                model = await tf.loadLayersModel(`indexeddb://${modelName}`); // IndexedDB からロード
                console.log("ローカルの学習済みモデルをロードしました");
            } else {
                console.log(`${url}/model.json`);
                model = await tf.loadLayersModel(`${url}/model.json`); // 外部モデルをロード
                console.log("サーバーからモデルをロードしました");
        }}
        document.getElementById("Attention").style.display = "none";
    } catch (error) {
        console.error("モデルのロードに失敗しました", error);
        document.getElementById("Attention").style.display = "block";
    }
}

// 2. 追加データを学習用に変換
async function addTrainingData(playerData, generatedMaterialIndex, who) {
    if (!model) {
        console.log("モデルがロードされていません");
        return;
    }

    // 入力データを取得
    var inputData = await convertToCount(playerData);
    var total = inputData.reduce(function(sum, element){return sum + element;}, 0);
    inputData.push(who);
    inputData.push(total*2 + Number(!who) + 1);
    console.log("学習用データ:", inputData);

    // データをTensorに変換
    const inputTensor = tf.tensor2d([inputData], [1, 26]);
    const outputTensor = tf.tensor2d([oneHotEncode(generatedMaterialIndex, model.outputShape[1])], [1, model.outputShape[1]]);

    // データセットに追加
    xs.push(inputTensor);
    ys.push(outputTensor);
    console.log("データを追加しました: クラス", generatedMaterialIndex);
}

// 3. モデルの追加学習
// コサイン類似度の計算関数
function cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        normA += vec1[i] ** 2;
        normB += vec2[i] ** 2;
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    return normA && normB ? dotProduct / (normA * normB) : 0;
}

// 🎯 **最も近い物質を取得する関数**
function findClosestMaterial(handVector) {
    let bestMatch = null;
    let bestSimilarity = -1;

    materials.forEach((material, index) => {
        let materialVec = Object.values(material.d); // 元素のベクトル化
        let similarity = cosineSimilarity(handVector, materialVec);

        if (similarity > bestSimilarity) {
            bestSimilarity = similarity;
            bestMatch = { index, similarity };
        }
    });

    return bestMatch;
}

// 3. モデルの追加学習
async function trainModel() {
    if (!model || xs.length === 0) {
        console.log("学習データが不足しています");
        return;
    }

    if (isTraining) {
        console.log("現在学習中です...");
        return;
    }

    isTraining = true;
    console.log("モデルの追加学習を開始...");

    // 🎯 **モデルのコンパイル（初期学習用）**
    model.compile({
        optimizer: tf.train.adam(0.002),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // 🎯 **モデルの出力サイズを確認**
    console.log("model.outputShape:", model.outputShape);

    if (!model.outputShape || model.outputShape.length < 2) {
        console.error("モデルの outputShape が不正です:", model.outputShape);
        return;
    }

    // 🎯 **データを Tensor に変換**
    const xTrain = tf.concat(xs);
    const yTrain = tf.concat(ys);

    // 🎯 **基本の学習（プレイヤーデータで学習）**
    await model.fit(xTrain, yTrain, {
        epochs: 3,
        batchSize: 32,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    console.log("手札に最も近い物質のデータを追加学習...");

    let adjustedXs = [];
    let adjustedYs = [];

    // 🎯 **エラー防止: numClasses にデフォルト値を設定**
    let numClasses = model.outputShape[1] || (materials ? materials.length : 10);
    
    if (!numClasses || isNaN(numClasses)) {
        console.error("numClasses が不正です:", numClasses);
        isTraining = false;
        return;
    }

    xs.forEach((handVector, index) => {
        // 🎯 **現在の手札に最も近い物質を探す**
        let closestMaterial = findClosestMaterial(handVector.dataSync());

        if (!closestMaterial) {
            console.warn(`手札 ${index} に対応する近い物質が見つかりません。スキップします。`);
            return;
        }

        let materialIndex = closestMaterial.index;

        console.log(`学習対象: 手札 ${index} → 近い物質: materials[${materialIndex}]`);

        // 🎯 **追加データの作成**
        let adjustedLabels = oneHotEncode(materialIndex, numClasses);
        adjustedYs.push(tf.tensor2d([adjustedLabels], [1, numClasses]));
        adjustedXs.push(handVector); // **元の入力データを再利用**
    });

    if (adjustedXs.length === 0 || adjustedYs.length === 0) {
        console.warn("追加学習用のデータが不足しているため、スキップします。");
        isTraining = false;
        return;
    }

    // 🎯 **追加学習用のデータを Tensor に変換**
    const xTrainSim = tf.concat(adjustedXs);
    const yTrainSim = tf.concat(adjustedYs);

    // 🎯 **モデルのコンパイル（追加学習用）**
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // 🎯 **最も近い物質のデータで追加学習**
    await model.fit(xTrainSim, yTrainSim, {
        epochs: 1,
        batchSize: 32,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    console.log("モデルの追加学習が完了しました");

    // 🎯 **メモリ解放**
    xTrain.dispose();
    yTrain.dispose();
    xTrainSim.dispose();
    yTrainSim.dispose();
    xs = [];
    ys = [];
    isTraining = false;

    await saveModel();
}



function CanCreateMaterial(material) {
    // 必要な元素リスト
    const requiredElements = material.d;

    // 使用可能な元素のカウント
    const availableElements = {};

    // 使用可能なカードをすべて統合
    const allCards = [...deck, ...p2_hand];

    // 各カードの元素をカウント
    allCards.forEach(card => {
        availableElements[card] = (availableElements[card] || 0) + 1;
    });

    // 必要な元素が揃っているか確認
    for (const element in requiredElements) {
        if (!availableElements[element] || availableElements[element] < requiredElements[element]) {
            return true; // 必要な元素が不足している 「不足していなかったら」なのでここで反転させておく
        }
    }

    return false; // 全ての必要な元素が揃っている
}

function getUsedMaterials() {
    // localStorage から "materials" のデータを取得
    let storedMaterials = localStorage.getItem("materials");

    // データが null, 空文字, 空オブジェクトの場合は処理しない
    if (!storedMaterials || storedMaterials === "{}") {
        console.log("No valid materials data found.");
        return {};
    }

    // JSON をパース
    let materials = JSON.parse(storedMaterials);

    // 1回以上作成された（値が1以上の）物質のみを抽出
    let usedMaterials = Object.fromEntries(
        Object.entries(materials).filter(([key, value]) => value > 0)
    );

    return usedMaterials;
}

function calculatePseudoProbabilities(materials) {
    let total = Object.values(materials).reduce((sum, value) => sum + value, 0);
    if (total === 0) return {}; // すべて 0 なら確率なし

    let probabilities = {};
    for (let key in materials) {
        probabilities[key] = materials[key] / total;
    }

    return probabilities;
}

function calculateWeightedProbabilities(probabilities, outputData) {
    let weightedProbabilities = {};

    // 共通するキーがあれば掛け算し * 100、なければ outputData*0.1 にする
    for (let key in outputData) {
        if (probabilities.hasOwnProperty(key)) {
            sumNs = new Int8Array(localStorage.getItem("sumNs"))
            weightedProbabilities[key] = (probabilities[key]*sumNs / (sumNs + 10) + outputData[key]) /2; //\frac{x}{x+c} という関数で0→0、∞→1となる関数。cで速さを調整可能。
        } else {
            weightedProbabilities[key] = outputData[key];
        }
    }

    return weightedProbabilities;
}

//推論
async function runModel(who) {
    if (!model) {
        console.log("モデルがロードされていません");
        return;
    }

    // 入力データ
    var inputData = await convertToCount();
    var total = inputData.reduce(function(sum, element){return sum + element;}, 0);
    inputData.push(who);
    inputData.push(total*2 + Number(!who) +1);

    inputData = tf.tensor2d([inputData], [1, 26]);

    // 推論実行
    const output = model.predict(inputData);
    let outputData = await output.data();

    recordCreatedMaterials = getUsedMaterials()
    pseudoProbability = calculatePseudoProbabilities(recordCreatedMaterials)
    console.log(pseudoProbability)

    let weightedResults = calculateWeightedProbabilities(pseudoProbability, outputData);
    console.log(weightedResults)


    // Math.max を使って最大値を取得
    var confidence = Math.max(...Object.values(weightedResults));

    // 最大値に対応するキーを検索
    var predictedClass = Object.keys(weightedResults).find(key => weightedResults[key] === confidence);


    while (await CanCreateMaterial(materials[predictedClass])) {
        // weightedResults から現在の predictedClass を削除
        delete weightedResults[predictedClass];
    
        if (Object.keys(weightedResults).length === 0) {
            console.log("作成できる候補がありません");
            return;
        }
    
        // Math.max を使って最大値を取得
        var confidence = Math.max(...Object.values(weightedResults));
    
        // 最大値に対応するキーを検索（数値型に変換）
        var predictedClass = Object.keys(weightedResults).find(key => weightedResults[key] === confidence);
    }
    

    // 結果を表示
    console.log(`推論結果: クラス ${predictedClass}, 信頼度: ${confidence}`);
    document.getElementById("predictResult").innerHTML = `予測結果：${materials[predictedClass].a}・信頼度：${confidence}`;


    return { predictedClass, confidence };
}


// 5. 学習済みモデルを IndexedDB に保存
async function saveModel() {
    if (!model) {
        console.log("モデルがロードされていません");
        return;
    }

    try {
        console.log(modelName)
        console.log(`indexeddb://${modelName}`)
        await model.save(`indexeddb://${modelName}`); // IndexedDB に保存
        console.log("学習済みモデルを IndexedDB に保存しました");
    } catch (error) {
        console.error("モデルの保存に失敗しました", error);
    }
}

// One-Hot エンコーディング関数
function oneHotEncode(index, numClasses) {
    const encoded = new Array(numClasses).fill(0);
    encoded[index] = 1;
    return encoded;
}

//　load materials
async function loadMaterials(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (!data.material || !Array.isArray(data.material)) {
            document.getElementById("Attention2").style.display = "inline";
            return [];
        }
        document.getElementById("Attention2").style.display = "none";
        return data.material;
    } catch (error) {
        console.error("Error fetching compounds:", error);  // Log the error to the console for debugging
        document.getElementById("Attention2").style.display = "inline";
        return []; // Return an empty array in case of error
    }
}



// main code
async function view_p2_hand() {
    const area = document.getElementById('p2_hand')
    p2_hand.forEach((elem, index) => {
        const image = document.createElement("img")
        image.src = imageCache[elementToNumber[elem]].src
        image.alt = elem
        image.style.padding = "5px"
        image.style.border = "1px solid #000"
        image.classList.add("selected")
        image.addEventListener("click", function() {
            const button = document.getElementById("ron_button")
            button.style.display = "none"
            if (time == "make") {
                this.classList.toggle("selected")
                if (this.classList.contains("selected")){
                    this.style.border = "1px solid #000"
                    this.style.padding = "5px"
                    p2_selected_card.splice(p2_selected_card.indexOf(this.alt),1)
                } else {
                    this.style.border = "5px solid #F00"
                    this.style.padding = "1px"
                    p2_selected_card.push(this.alt)
                }}
            if (turn == "p2" && time == "game") {
                dropped_cards_p2.push(this.alt)
                const img = document.createElement("img")
                img.alt = this.alt
                img.src = imageCache[elementToNumber[this.alt]].src
                img.style.border = "1px solid #000"
                document.getElementById("dropped_area_p2").appendChild(img)
                this.classList.remove("selected")
                this.classList.add("selected")
                let newElem = drawCard()
                this.src = imageCache[elementToNumber[newElem]].src
                this.alt = newElem
                this.style.padding = "5px"
                this.style.border = "1px solid #000"
                p2_hand[index] = newElem
                turn = "p1"
                setTimeout(() => {p1_action()},500)
            }
        })
        area.appendChild(image)
    })
}

async function view_p1_hand() {
    const area = document.getElementById('p1_hand')
    p1_hand.forEach((elem, index) => {
        const image = document.createElement("img")
        image.src = imageCache[0].src
        image.alt = "相手の手札"
        image.style.padding = "5px"
        image.style.border = "1px solid #000"
        image.classList.add("selected")
        area.appendChild(image)
    })
}

async function search(components) {
    return materials.find(material => {
        for (const element in components) {
            if (!material.d[element] || material.d[element] !== components[element]) {
                return false;
            }
        }
        for (const element in material.d) {
            if (!components[element]) {
                return false;
            }
        }
        return true;
    }) || materials[0];
}

async function p1_make(predictedMaterialP2) {
    const makeable_material = await search_materials(arrayToObj(p1_hand));

    // 作れる物質がない場合は "なし" を返す
    if (!makeable_material || makeable_material.length === 0) {
        return [{
            "a": "なし",
            "b": "なし",
            "c": 0,
            "d": {},
            "e": []
        }];
    }

    // ポイントが高い順にソート
    makeable_material.sort((a, b) => b.c - a.c);

    return makeable_material;
}

async function p2_make() {
    // ボタンの表示を変更
    document.getElementById("generate_button").style.display = "none";
    const button = document.getElementById("done_button");
    button.style.display = "inline";

    // 以前のイベントリスナーを削除
    button.replaceWith(button.cloneNode(true));
    const newButton = document.getElementById("done_button");

    // ボタンクリックを待機
    return new Promise((resolve) => {
        newButton.addEventListener("click", function () {
            const p2_make_material = search(arrayToObj(p2_selected_card));
            resolve(p2_make_material);
        });
    });
}


async function get_dora() {
    return element[Math.round(Math.random()*23)]
}

async function incrementMaterialCount(material) {
    // localStorage から "materials" キーのデータを取得
    let materialsData = localStorage.getItem("materials");

    // JSONをパース（データがない場合は空のオブジェクトを設定）
    let materials = materialsData ? JSON.parse(materialsData) : {};

    // 指定された material の値を1増やす（存在しない場合は初期値1）
    materials[material] = (materials[material] || 0) + 1;

    // 更新したオブジェクトをJSONに変換してlocalStorageに保存
    localStorage.setItem("materials", JSON.stringify(materials));
    var sumNs = new Int8Array(localStorage.getItem("sumNs"))
    localStorage.setItem("sumNs", (sumNs)+1)
}


async function done(who, isRon = false) {

    const p2_make_material = await p2_make();
    predictedMaterialP2 = await runModel(who=="p1" ? 0:1)
    const p1_make_material = await p1_make(predictedMaterialP2);

    dora = await get_dora();
    console.log(`ドラ: ${dora}`);
    
    let thisGame_p2_point = p2_make_material.c;
    let thisGame_p1_point = p1_make_material[0].c;

    // 有利な生成物の場合のボーナス
    if (Boolean(p2_make_material.e.includes(p1_make_material[0].b))) {
        thisGame_p2_point *= (1.5 + Math.random() / 2);
    } else if (Boolean(p1_make_material[0].e.includes(p2_make_material.b))) {
        thisGame_p1_point *= (1.5 + Math.random() / 2);
    }

    // 役の中にドラが含まれる場合のボーナス
    if (Boolean(Object.keys(p2_make_material.d).includes(dora))) {
        thisGame_p2_point *= 1.5;
    } else if (Boolean(Object.keys(p1_make_material[0].d).includes(dora))) {
        thisGame_p1_point *= 1.5;
    }

    // **ロン時のボーナス**
    if (isRon) {
        who == "p2" ? thisGame_p2_point /= 1.2 : thisGame_p1_point /= 1.2
    }

    who == "p2" ? thisGame_p1_point /= 1.5 : thisGame_p2_point /= 1.5;

    // 小数点以下を四捨五入
    thisGame_p2_point = Math.round(thisGame_p2_point);
    thisGame_p1_point = Math.round(thisGame_p1_point);

    // 得点を更新
    p1_point += await thisGame_p1_point;
    p2_point += await thisGame_p2_point;

    // 画面に反映
    document.getElementById("p2_point").innerHTML += `+${thisGame_p2_point}`;
    document.getElementById("p1_point").innerHTML += `+${thisGame_p1_point}`;
    document.getElementById("p2_explain").innerHTML = `生成物質：${p2_make_material.a}, 組成式：${p2_make_material.b}`;
    document.getElementById("p1_explain").innerHTML = `生成物質：${p1_make_material[0].a}, 組成式：${p1_make_material[0].b}`;

    //モデルを学習
    let playerData = convertToCount(dropped_cards_p2)
    let generatedMaterialIndex = p2_make_material.f
    await addTrainingData(playerData, generatedMaterialIndex, who=="p1" ? 0:1);
    await trainModel();

    await incrementMaterialCount(p2_make_material.a)

    // 勝者判定
    const winner = await win_check();
    
    document.getElementById("done_button").style.display = "none";
    const button = document.getElementById("nextButton");
    button.style.display = "inline";

    if (!winner) {
        console.log("次のゲーム");
        numTurn += 1;
        button.textContent = "次のゲーム";
        button.addEventListener("click", function () {
            resetGame();
            button.style.display = "none"
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);
        });
    } else {
        console.log("ゲーム終了");
        button.textContent = "ラウンド終了";
        button.addEventListener("click", function () {
            returnToStartScreen()
            p1_point = 0;
            p2_point = 0;
            numTurn = 0;
            resetGame();
            button.style.display = "none"
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);
        });
    }
}

async function win_check() {
    return Math.abs(p1_point - p2_point) >= WIN_POINT ? p1_point>p2_point ? "p1": "p2" : numTurn >= WIN_TURN ? p1_point>p2_point ? "p1": "p2" : null
}

async function p1_exchange(targetElem) {
    // Select a random card index from p1_hand// TODO: from AI.js
    dropped_cards_p1.push(p1_hand[targetElem])
    var exchange_element = p1_hand[targetElem]
    // Ensure the target card exists and is valid
    if (!p1_hand[targetElem]) {
        console.error("Invalid target element in p1_hand.")
        return
    }
    // Create a new image for the dropped card area
    const newImg = document.createElement("img")
    newImg.src = imageCache[elementToNumber[p1_hand[targetElem]]].src
    newImg.style.border = "1px solid #000"
    document.getElementById("dropped_area_p1").appendChild(newImg)
    // Update the player's hand with a new element
    const img = document.querySelectorAll("#p1_hand img")[targetElem]
    if (!img) {
        console.error("Image element not found in p1_hand.")
        return
    }
    // Select a new random element and replace the target card
    const newElem = drawCard()
    p1_hand[targetElem] = newElem
    // Update the image element's appearance
    img.src = imageCache[0].src
    img.alt = newElem
    img.style.padding = "5px"
    img.style.border = "1px solid #000"
    // Remove and reapply the 'selected' class to reset the state
    img.classList.remove("selected")
    img.classList.add("selected")
    // Switch the turn to "p1"
    turn = "p2"
    checkRon(exchange_element);
}

async function p1_action() {
    if (turn !== "p1" || p1_is_acting) {
        return;  // すでに行動中なら何もしない
    }
    p1_is_acting = true;  // 行動開始
    
    // フィルタリング
    const highPointMaterials = materials.filter(material => material.c > 20);
    
    // 最適な物質を選択
    const sortedMaterials = highPointMaterials.sort((a, b) => {
        let aMatchCount = Object.keys(a.d).reduce((count, elem) => count + Math.min(p1_hand.filter(e => e === elem).length, a.d[elem]), 0);
        let bMatchCount = Object.keys(b.d).reduce((count, elem) => count + Math.min(p1_hand.filter(e => e === elem).length, b.d[elem]), 0);
        return bMatchCount - aMatchCount || b.c - a.c;
    });

    const targetMaterial = sortedMaterials[0];

    if (!targetMaterial) {
        p1_exchange(Math.floor(Math.random() * p1_hand.length));
    } else {
        let canMake = true;
        for (const element in targetMaterial.d) {
            if (!p1_hand.includes(element) || p1_hand.filter(e => e === element).length < targetMaterial.d[element]) {
                canMake = false;
                break;
            }
        }

        if (canMake && targetMaterial.c > 20) {
            time = "make";
            await done("p1");
        } else {
            let unnecessaryCards = p1_hand.filter(e => {
                return !(e in targetMaterial.d) || p1_hand.filter(card => card === e).length > targetMaterial.d[e];
            });

            if (unnecessaryCards.length > 0) {
                let cardToExchange = unnecessaryCards[Math.floor(Math.random() * unnecessaryCards.length)];
                p1_exchange(p1_hand.indexOf(cardToExchange));
            } else {
                time = "make"
                done("p1");
            }
        }
    }
    
    turn = "p2";
    p1_is_acting = false;
}



//便利系関数
function arrayToObj(array) {
    let result = {}
    array.forEach(item => {
        if (result[item]) {
            result[item]++
        } else {
            result[item] = 1
        }
    })
    return result
}

function shuffle(array) {
    let currentIndex = array.length;
  
    // While there remain elements to shuffle...
    while (currentIndex != 0) {
  
      // Pick a remaining element...
      let randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    };

    return array;
}

function drawCard() {
    return deck.length > 0 ? deck.pop() : (time = "make", done("no-draw"));
}

async function search_materials(components) {
    return materials.filter(material => {
        for (const element in material.d) {
            if (!components[element] || material.d[element] > components[element]) {
                return false;
            };
        };
        return true;
    });
}

function random_hand() {
    for (let i = 0; i < card_num; i++) {
        p1_hand.push(drawCard());
        p2_hand.push(drawCard());
    };
}

document.getElementById("generate_button").addEventListener("click", function () {
    if (turn == "p2") {
        time = "make"
        const newRonButton = document.getElementById("ron_button");
        newRonButton.style.display = "none";
        done("p2");
    }
})

function resetGame() {
    p1_hand = [];
    p2_hand = [];
    dropped_cards_p1 = [];
    dropped_cards_p2 = [];
    p1_selected_card = [];
    p2_selected_card = [];
    time = "game";
    turn = Math.random() <= 0.5 ? "p1" : "p2";
    numTurn = 1;  // ターンカウントをリセット

    document.getElementById("p1_point").innerHTML = `ポイント：${p1_point}`;
    document.getElementById("p1_explain").innerHTML = "　";
    document.getElementById("p2_point").innerHTML = `ポイント：${p2_point}`;
    document.getElementById("p2_explain").innerHTML = "　";
    document.getElementById("predictResult").innerHTML = "　";

    document.getElementById("generate_button").style.display = "inline";
    document.getElementById("done_button").style.display = "none";
    document.getElementById("nextButton").style.display = "none";

    deck = [...elements, ...elements];
    deck = shuffle(deck);

    const p1_hand_element = document.getElementById("p1_hand");
    const p2_hand_element = document.getElementById("p2_hand");
    p1_hand_element.innerHTML = "";
    p2_hand_element.innerHTML = "";

    const dropped_area_p1_element = document.getElementById("dropped_area_p1");
    const dropped_area_p2_element = document.getElementById("dropped_area_p2");
    dropped_area_p1_element.innerHTML = "";
    dropped_area_p2_element.innerHTML = "";

    random_hand();
    view_p1_hand();
    view_p2_hand();

    if (turn === "p1") {
        setTimeout(() => p1_action(), 500);
    }
}

function preloadImages() {
    let imageNumbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 29, 30, 53];

    imageNumbers.forEach(num => {
        let img = new Image();
        img.src = `../images/${num}.webp`;
        imageCache[num] = img;
    });
}

async function init_json() {
    materials = await loadMaterials("https://kurorosuke.github.io/compounds/obf_extended_min.json");
}



async function checkRon(droppedCard) {
    // P2のロン判定
    const possibleMaterialsP2 = await search_materials(arrayToObj([...p2_hand, droppedCard]));
    const validMaterialsP2 = possibleMaterialsP2.filter(material => material.d[droppedCard]);
    if (validMaterialsP2.length > 0) {
        const ronButton = document.getElementById("ron_button");
        ronButton.style.display = "inline";
        ronButton.replaceWith(ronButton.cloneNode(true));
        const newRonButton = document.getElementById("ron_button");

        newRonButton.addEventListener("click", function () {
            newRonButton.style.display = "none";
            const dropped = document.querySelectorAll("#dropped_area_p1 img");
            const selectCard = dropped[dropped.length - 1];
            selectCard.style.border = "2px solid red";
            selectCard.style.padding = "1px";
            p2_selected_card = [droppedCard];
            time = "make";
            // 捨て牌一覧の最後の要素を取得し、赤枠を付ける
            const DroppedCards = document.getElementById("dropped_area_p1").children
            const lastDiscard = DroppedCards[DroppedCards.length - 1]
            lastDiscard.style.border = "2px solid f00";
            done("p2", true);
        });
    }

    // P1のロン判定（捨てられたカードを含める）
    const possibleMaterialsP1 = await search_materials(arrayToObj([...p1_hand, droppedCard]));
    const validMaterialsP1 = possibleMaterialsP1.filter(material => ((material.c >= 70) && material.d[droppedCard]));

    if (validMaterialsP1.length > 0) {
        // **P1の手札に捨てたカードがもうない可能性があるため、戻す**
        p1_hand.push(droppedCard);
        // P1のロン処理のため、ロンに使うカードを選択
        p1_selected_card = [droppedCard];
        // `time` を "make" に変更
        time = "make";
        // P1のロン処理を実行
        done("p1", true);
    }
}

function updateGeneratedMaterials(materialName) {
    if (!materialName || materialName === "なし") return;

    // LocalStorage からデータを取得（なければ空のオブジェクト）
    let generatedMaterials = JSON.parse(localStorage.getItem("generatedMaterials")) || {};

    // 物質のカウントを更新
    if (generatedMaterials[materialName]) {
        generatedMaterials[materialName] += 1;
    } else {
        generatedMaterials[materialName] = 1;
    }

    // LocalStorage に保存
    localStorage.setItem("generatedMaterials", JSON.stringify(generatedMaterials));
}

//設定画面
function openWinSettings() {
    document.getElementById("winSettingsModal").style.display = "block";
}
async function saveWinSettings() {
    let winPointInput = parseInt(document.getElementById("winPointInput").value, 10);
    let winTurnInput = parseInt(document.getElementById("winTurnInput").value, 10);

    if (isNaN(winPointInput) || winPointInput < 1) {
        alert("WIN_POINT は 1 以上の数値を入力してください。");
        return;
    }
    if (isNaN(winPointInput) || winPointInput > 999) {
        alert("WIN_POINT の最大値は 999 です。");
        return;
    }
    if (isNaN(winTurnInput) || winTurnInput < 1) {
        alert("WIN_TURN は 1 以上の数値を入力してください。");
        return;
    }

    let compoundsValue = document.getElementById("compoundsSelection").value;
    if (compoundsValue != "url") {
        var compoundsURL = `https://kurorosuke.github.io/compounds/${compoundsValue}.json`;
    } else {
        var compoundsURL = document.getElementById("compoundsURL").value;
    }
    materials = await loadMaterials(compoundsURL);
    
    var modelSelect = document.getElementById("modelSelection").value;
    if (modelSelect!="new"){
        modelURL = `https://kurorosuke.github.io/AI_models/${modelSelect}`;
    } else {
        modelURL = document.getElementById("modelURL").value;
    }
    model = loadModel(modelURL);

    WIN_POINT = winPointInput;
    WIN_TURN = winTurnInput;
    closeWinSettings();
}
function closeWinSettings() {
    document.getElementById("winSettingsModal").style.display = "none";
}
document.getElementById("setting_icon").addEventListener("click", function() {
    document.getElementById("winSettingsModal").style.display = "inline"
})


//ヒント
async function findMostPointMaterial() {
    const possibleMaterials = await search_materials(arrayToObj(p2_hand));
    
    if (possibleMaterials.length === 0) {
        console.log("p2_hand 内で作成可能な物質はありません。");
    } else {
        const highestMaterial = possibleMaterials.reduce((max, material) => 
            material.c > max.c ? material : max, possibleMaterials[0]);
        console.log(`p2_hand 内で最もポイントが高い物質: ${highestMaterial.a} (ポイント: ${highestMaterial.c})`);
    }
}

function initializeMaterials() {
    // localStorage に "materials" が存在しない場合
    if (!localStorage.getItem("materials")) {
        // materials 内の各オブジェクトの a キーの値をキーとし、値を 0 にするオブジェクトを作成
        let initialMaterials = {};
        materials.forEach(item => {
            initialMaterials[item.a] = 0;
        });

        // 作成したオブジェクトを localStorage に保存
        localStorage.setItem("materials", JSON.stringify(initialMaterials));
    }
    if (!localStorage.getItem("sumNs")) {
        localStorage.setItem("sumNs", 0);
    }
}

document.addEventListener('DOMContentLoaded', function () {
    preloadImages()
    init_json()
    loadModel()
    initializeMaterials()
    deck = [...elements, ...elements]
    deck = shuffle(deck)
    random_hand()
    view_p1_hand()
    view_p2_hand()
    addOptions()
    turn = Math.random()>=0.5 ? "p1" : "p2"
    if (turn == "p1") {p1_action()}
})

function returnToStartScreen() {
    document.getElementById("startScreen").style.display = "flex";
    document.getElementById("p1_area").style.display = "none";
    document.getElementById("dropped_area_p1").style.display = "none";
    document.getElementById("dropped_area_p2").style.display = "none";
    document.getElementById("p2_area").style.display = "none";
    document.getElementById("gameRuleButton").style.display = "block";
}
document.getElementById("startButton").addEventListener("click", function() {
    document.getElementById("startScreen").style.display = "none";
    document.getElementById("p1_area").style.display = "block";
    document.getElementById("dropped_area_p1").style.display = "block";
    document.getElementById("dropped_area_p2").style.display = "block";
    document.getElementById("p2_area").style.display = "block";
    document.getElementById("gameRuleButton").style.display = "none";
});


function showRules() {
    document.getElementById("rulesModal").style.display = "block";
}

function closeRules() {
    document.getElementById("rulesModal").style.display = "none";
}

document.getElementById("closeRulesButton").addEventListener("click", closeRules);

// モーダル外をクリックした場合に閉じる
window.onclick = function(event) {
    const modal = document.getElementById("rulesModal");
    if (event.target === modal) {
        closeRules();
    }
};

function showInputTag() {
    if (document.getElementById("compoundsSelection").value == "url"){
        document.getElementById("compoundsURL").style.display = "inline";
    } else {
        document.getElementById("compoundsURL").style.display = "none";
    }
}

function showModelInputTag() {
    if (document.getElementById("modelSelection").value == "new"){
        document.getElementById("modelURL").style.display = "inline";
    } else {
        document.getElementById("modelURL").style.display = "none";
    }
}

async function getModelNames() {
    try {
        const models = await tf.io.listModels();
        const modelNames = Object.keys(models).map(key => key.replace('indexeddb://', ''));
        console.log(modelNames);
        return modelNames;
    } catch (error) {
        console.error("モデル名の取得に失敗しました", error);
        return [];
    }
}

async function addOptions() {
    let models = await getModelNames();
    const Selection = document.getElementById("modelModals")
    models.forEach(elem => {
        const newOption = document.createElement("div");
        newOption.className = "modelModal";
        newOption.id = elem
        newOption.text  = elem;
        const title = document.createElement("p");
        title.textContent = elem;
        newOption.appendChild(title);

        let selectButton = document.createElement("button");
        selectButton.textContent = "選択";
        selectButton.id = newOption.id;
        selectButton.onclick = function() { selectModelOnSetting(this.id); };
        
        // 削除ボタン
        let deleteButton = document.createElement("button");
        deleteButton.textContent = "削除";
        selectButton.id = newOption.id;
        deleteButton.onclick = function() { removeModelOnSetting(this.id); };
        
        // 初期化ボタン
        let resetButton = document.createElement("button");
        resetButton.textContent = "初期化";
        resetButton.onclick = function() { console.log("初期化が実行されました"); };
        
        // 保存ボタン
        let saveButton = document.createElement("button");
        saveButton.textContent = "保存";
        saveButton.onclick = function() { console.log("保存が実行されました"); };

        // 要素をモーダルに追加
        newOption.appendChild(title);
        newOption.appendChild(selectButton);
        newOption.appendChild(saveButton);
        newOption.appendChild(deleteButton);
        newOption.appendChild(resetButton);
        

        Selection.appendChild(newOption)
    })
}

function cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        normA += vec1[i] ** 2;
        normB += vec2[i] ** 2;
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    return normA && normB ? dotProduct / (normA * normB) : 0;
}

function pseudoCosVec(materialNum1, materialNum2) {
    const vec1 = convertToVector(materials[materialNum1].d, element);
    const vec2 = convertToVector(materials[materialNum2].d, element);
    console.log(vec1, vec2)
    const cos = cosineSimilarity(vec1, vec2)
    return cos
}

// 物質をベクトル化
function convertToVector(material, elementDict) {
    return elementDict.map(el => material[el] || 0);
}

function showModelDetail() {
    document.getElementById("modelModals").style.display = "inline";
    document.getElementById("buttonModal").style.display = "inline";
}

let selectingModel;
function selectModelOnSetting(modelName) {
    selectingModel = modelName;
    const modelDivs = document.querySelectorAll("#modelModals div");
    modelDivs.forEach(elem => {
        elem.style.background = "white";
    })
    document.getElementById(modelName).style.background = "pink";
}

function applyModalSetting() {
    closeModelModal();
    document.getElementById("Attention2").innerHTML = "do this";
    loadModel("notNull",selectingModel);
}

function closeModelModal() {
    document.getElementById("modelModals").style.display = "none";
    document.getElementById("buttonModal").style.display = "none";
    document.getElementById("winSettingsModal").style.display = "none";
}
